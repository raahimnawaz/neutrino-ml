"""Task A, representation 3 -- an event-level Graph Neural Network.

Instead of the hand-engineered feature table the XGBoost/logreg models use, this represents
each event as a small **graph**: one node per reconstructed object (the lepton, each jet,
and the MET), fully connected. A GNN learns directly from the objects and their relations,
with no hand-crafted event summary -- the "particle-level representation" from the project's
original plan.

Honest expectation: this final state has only ~2-6 objects per event, so the graphs are tiny
and the GNN is *not* expected to beat the gradient-boosted trees (few-object structured data
favours trees). The value is the representation comparison and the technique itself -- the same
graph/point-cloud machinery used in modern perception. The model is device-aware (cuda / mps /
cpu), so it uses a GPU/MPS if one is present.

Usage:
    python -m src.gnn_taskA            # real data if present, else synthetic; caps events for CPU
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

from src import physics, read_root, selection, weights
from src.pipeline import _abspath, _load_config, _real_data_files, REPO_ROOT

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "saved_models")
NODE_DIM = 7            # [pt/100, eta, sin phi, cos phi, is_lepton, is_jet, is_met]
MAX_EVENTS = 60000     # cap total events so CPU training stays tractable


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _node(pt, eta, phi, is_lep, is_jet, is_met):
    """One node feature vector. sin/cos(phi) avoid the phi wrap discontinuity."""
    return [pt / 100.0, eta, np.sin(phi), np.cos(phi),
            float(is_lep), float(is_jet), float(is_met)]


def event_to_graph(lep_pt, lep_eta, lep_phi, jet_pts, jet_etas, jet_phis,
                   met_et, met_phi, label, weight) -> Data:
    """Build one fully-connected event graph: lepton + jets + MET nodes."""
    nodes = [_node(lep_pt, lep_eta, lep_phi, 1, 0, 0)]
    for jp, je, jf in zip(jet_pts, jet_etas, jet_phis):
        nodes.append(_node(jp, je, jf, 0, 1, 0))
    nodes.append(_node(met_et, 0.0, met_phi, 0, 0, 1))  # MET: no eta -> 0

    x = torch.tensor(nodes, dtype=torch.float)
    n = x.shape[0]
    # fully-connected edges (no self-loops); small graphs so this is cheap
    idx = [[i, j] for i in range(n) for j in range(n) if i != j]
    edge_index = torch.tensor(idx, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.float),
                w=torch.tensor([weight], dtype=torch.float))


def build_graphs(signal_files, bkg_files, cfg):
    """Read + select real events and turn each into an event graph."""
    graphs = []
    for files, label in [(signal_files, 1), (bkg_files, 0)]:
        events = read_root.read_events(files, tree="mini")
        mask, _ = selection.select(events, cfg)
        mask = np.asarray(mask, dtype=bool)
        w_all = weights.event_weights(events, float(cfg["data"]["lumi_pb"]))[mask]

        lep_pt = events["lep_pt"][mask]; lep_eta = events["lep_eta"][mask]
        lep_phi = events["lep_phi"][mask]
        jet_pt = events["jet_pt"][mask]; jet_eta = events["jet_eta"][mask]
        jet_phi = events["jet_phi"][mask]
        met_et = np.asarray(events["met_et"], float)[mask]
        met_phi = np.asarray(events["met_phi"], float)[mask]

        for i in range(int(mask.sum())):
            graphs.append(event_to_graph(
                float(np.asarray(lep_pt[i])[0]), float(np.asarray(lep_eta[i])[0]),
                float(np.asarray(lep_phi[i])[0]),
                np.asarray(jet_pt[i], float), np.asarray(jet_eta[i], float),
                np.asarray(jet_phi[i], float),
                met_et[i], met_phi[i], label, float(w_all[i])))
    return graphs


class EventGNN(nn.Module):
    """Two GraphSAGE layers + mean pooling + linear head -> one logit per graph."""

    def __init__(self, in_dim=NODE_DIM, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


def run(signal_files, bkg_files, cfg, epochs=25):
    torch.manual_seed(int(cfg["train"]["random_state"]))
    device = _device()
    print(f"[gnn] device: {device}")

    graphs = build_graphs(signal_files, bkg_files, cfg)
    if len(graphs) > MAX_EVENTS:
        rng = np.random.default_rng(int(cfg["train"]["random_state"]))
        graphs = [graphs[i] for i in rng.choice(len(graphs), MAX_EVENTS, replace=False)]
    y_all = np.array([int(g.y.item()) for g in graphs])
    print(f"[gnn] {len(graphs)} event graphs ({y_all.sum()} signal / {(y_all==0).sum()} bkg)")

    tr_idx, te_idx = train_test_split(np.arange(len(graphs)), test_size=0.3,
                                      random_state=int(cfg["train"]["random_state"]),
                                      stratify=y_all)
    train_set = [graphs[i] for i in tr_idx]
    test_set = [graphs[i] for i in te_idx]
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    model = EventGNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logit = model(batch)
            # weighted BCE; |w| for training (negative NLO weights, as with the trees)
            loss = (loss_fn(logit, batch.y) * batch.w.abs()).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * batch.num_graphs
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[gnn] epoch {ep:2d}  train loss {tot/len(train_set):.4f}")

    # --- evaluate: weighted AUC on the held-out test set -----------------------------
    model.eval()
    scores, ys, ws = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
            ys.append(batch.y.cpu().numpy())
            ws.append(batch.w.cpu().numpy())
    scores = np.concatenate(scores); ys = np.concatenate(ys); ws = np.abs(np.concatenate(ws))
    auc = roc_auc_score(ys, scores, sample_weight=ws)

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "event_gnn.pt"))
    print("\n" + "=" * 60)
    print("  neutrino-ml  |  Task A representation 3 (event GNN)")
    print("=" * 60)
    print(f"  GNN weighted ROC AUC:      {auc:.4f}")
    print(f"  (XGBoost engineered-feature AUC for reference: 0.825)")
    print(f"  saved model -> {os.path.join(MODELS_DIR, 'event_gnn.pt')}")
    print("=" * 60)
    return {"auc": float(auc)}


def main():
    cfg = _load_config(os.path.join(REPO_ROOT, "config.yaml"))
    root_dir = _abspath(cfg["data"]["root_dir"])
    sig, bkg = _real_data_files(cfg, root_dir)
    if sig and bkg:
        print(f"[gnn] using REAL ATLAS Open Data: {len(sig)} signal + {len(bkg)} bkg files")
    else:
        sig = sorted(glob.glob(os.path.join(root_dir, "sample_signal*.root")))
        bkg = sorted(glob.glob(os.path.join(root_dir, "sample_bkg*.root")))
        if not sig or not bkg:
            raise FileNotFoundError("No inputs. Run scripts/download_data.py or tests/make_sample.py")
        print(f"[gnn] using SYNTHETIC fixtures")
    run(sig, bkg, cfg)


if __name__ == "__main__":
    main()
