"""Task B via a GNN -- predict *where the neutrino is* with geometric/topological deep learning.

Each W -> l nu event becomes a small graph (lepton + jets + MET as nodes, fully connected);
a GraphSAGE network regresses the neutrino longitudinal momentum ``pz`` straight from that
object-graph -- no hand-engineered event summary. This is the "topological method to predict
where the neutrino is": the model learns on the event's topology/relations.

Same honest supervision as ``task_b_regression`` (there is no truth-level ``pz`` in the data):
the target is the analytic W-mass ``pz`` (smaller-|pz| root) on the ~59% of events that have a
real solution. We train on a subset of those, validate on held-out solvable events, and apply
to the ~41% with no analytic solution -- and compare head-to-head with the tabular XGBoost
regressor (RMSE 21.6 GeV) and the analytic fallback.

Usage:
    python -m src.gnn_taskB
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

from src import physics, read_root, selection
from src.gnn_taskA import _device
from src.pipeline import _abspath, _load_config, _real_data_files, REPO_ROOT

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "saved_models")
M_W = 80.379
NODE_DIM = 8          # [pt/100, eta, sin phi, cos phi, pz/100, is_lep, is_jet, is_met]
PZ_SCALE = 100.0      # target/feature scale so pz (±100s of GeV) trains stably


def _node(pt, eta, phi, pz, is_lep, is_jet, is_met):
    return [pt / 100.0, eta, np.sin(phi), np.cos(phi), pz / PZ_SCALE,
            float(is_lep), float(is_jet), float(is_met)]


def _graph(lep, jets, met, target_pz) -> Data:
    lp, le, lf, lpz = lep
    nodes = [_node(lp, le, lf, lpz, 1, 0, 0)]
    for jp, je, jf, jpz in jets:
        nodes.append(_node(jp, je, jf, jpz, 0, 1, 0))
    met_et, met_phi = met
    nodes.append(_node(met_et, 0.0, met_phi, 0.0, 0, 0, 1))  # MET: pz unknown -> 0
    x = torch.tensor(nodes, dtype=torch.float)
    n = x.shape[0]
    idx = [[i, j] for i in range(n) for j in range(n) if i != j]
    edge_index = torch.tensor(idx, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index,
                y=torch.tensor([target_pz / PZ_SCALE], dtype=torch.float))


def build(signal_files, cfg):
    """Return (graphs, has_real mask, components dict) for all selected W->l nu events."""
    ev = read_root.read_events(signal_files, tree="mini")
    mask = np.asarray(selection.select(ev, cfg)[0], dtype=bool)
    lep_pt, lep_eta = ev["lep_pt"][mask], ev["lep_eta"][mask]
    lep_phi, lep_E = ev["lep_phi"][mask], ev["lep_E"][mask]
    jet_pt, jet_eta, jet_phi = ev["jet_pt"][mask], ev["jet_eta"][mask], ev["jet_phi"][mask]
    met_et = np.asarray(ev["met_et"], float)[mask]
    met_phi = np.asarray(ev["met_phi"], float)[mask]

    graphs, has_real = [], []
    comp = {k: [] for k in ("lpx", "lpy", "lpz", "lE", "mx", "my", "analytic_pz")}
    for i in range(int(mask.sum())):
        lp = float(np.asarray(lep_pt[i])[0]); le = float(np.asarray(lep_eta[i])[0])
        lf = float(np.asarray(lep_phi[i])[0]); lE = float(np.asarray(lep_E[i])[0])
        lpx, lpy, lpz = lp * np.cos(lf), lp * np.sin(lf), lp * np.sinh(le)
        mx, my = met_et[i] * np.cos(met_phi[i]), met_et[i] * np.sin(met_phi[i])

        pz1, pz2 = physics.neutrino_pz(lpx, lpy, lpz, lE, mx, my, mW=M_W)
        tgt = pz1 if abs(pz1) <= abs(pz2) else pz2
        m = physics.reconstruct_w_mass(lpx, lpy, lpz, lE, mx, my, tgt)
        has_real.append(abs(m - M_W) < 0.5)

        jp = np.asarray(jet_pt[i], float); je = np.asarray(jet_eta[i], float)
        jf = np.asarray(jet_phi[i], float)
        jets = [(jp[k], je[k], jf[k], jp[k] * np.sinh(je[k])) for k in range(len(jp))]
        graphs.append(_graph((lp, le, lf, lpz), jets, (met_et[i], met_phi[i]), tgt))
        for k, v in zip(comp, (lpx, lpy, lpz, lE, mx, my, tgt)):
            comp[k].append(v)
    comp = {k: np.array(v) for k, v in comp.items()}
    return graphs, np.array(has_real), comp


class GNNRegressor(nn.Module):
    def __init__(self, in_dim=NODE_DIM, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


def run(signal_files, cfg, epochs=30):
    torch.manual_seed(int(cfg["train"]["random_state"]))
    device = _device()
    print(f"[gnn-reg] device: {device}")

    graphs, has_real, comp = build(signal_files, cfg)
    print(f"[gnn-reg] {len(graphs)} events | analytic real solutions: "
          f"{has_real.sum()} ({100*has_real.mean():.1f}%)  no solution: {(~has_real).sum()}")

    idx = np.arange(len(graphs))
    solv = idx[has_real]
    uns = idx[~has_real]
    tr, te = train_test_split(solv, test_size=0.3, random_state=int(cfg["train"]["random_state"]))

    train_loader = DataLoader([graphs[i] for i in tr], batch_size=256, shuffle=True)
    test_loader = DataLoader([graphs[i] for i in te], batch_size=512)
    uns_loader = DataLoader([graphs[i] for i in uns], batch_size=512)

    model = GNNRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()  # Huber: robust to the pz tails

    for ep in range(epochs):
        model.train(); tot = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward(); opt.step()
            tot += loss.item() * batch.num_graphs
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[gnn-reg] epoch {ep:2d}  train loss {tot/len(tr):.4f}")

    def predict(loader):
        model.eval(); out = []
        with torch.no_grad():
            for batch in loader:
                out.append(model(batch.to(device)).cpu().numpy() * PZ_SCALE)
        return np.concatenate(out) if out else np.array([])

    # validation vs analytic pz on held-out solvable events
    pred_te = predict(test_loader)
    rmse = float(np.sqrt(np.mean((pred_te - comp["analytic_pz"][te]) ** 2)))
    print(f"[gnn-reg] held-out RMSE vs analytic pz: {rmse:.1f} GeV  "
          f"(XGBoost regressor: 21.6 GeV)")

    # apply to the no-solution events; reconstruct the W mass
    pred_uns = predict(uns_loader)
    m_analytic = physics.reconstruct_w_mass(comp["lpx"][uns], comp["lpy"][uns], comp["lpz"][uns],
                                            comp["lE"][uns], comp["mx"][uns], comp["my"][uns],
                                            comp["analytic_pz"][uns])
    m_gnn = physics.reconstruct_w_mass(comp["lpx"][uns], comp["lpy"][uns], comp["lpz"][uns],
                                       comp["lE"][uns], comp["mx"][uns], comp["my"][uns], pred_uns)
    print(f"[gnn-reg] no-solution reconstructed W mass: analytic median={np.median(m_analytic):.1f} "
          f"-> GNN median={np.median(m_gnn):.1f} GeV (XGBoost was 95.8; floor = mT > mW)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = np.percentile(np.abs(comp["analytic_pz"][te]), 99)
    ax1.scatter(comp["analytic_pz"][te], pred_te, s=3, alpha=0.2, color="#9467bd")
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("analytic $p_z^\\nu$ [GeV]"); ax1.set_ylabel("GNN $p_z^\\nu$ [GeV]")
    ax1.set_title(f"GNN validation on solvable events\nRMSE = {rmse:.1f} GeV")
    ax2.hist(np.clip(m_analytic, 0, 200), bins=50, range=(0, 200), alpha=0.55,
             label="analytic fallback", color="#d62728")
    ax2.hist(np.clip(m_gnn, 0, 200), bins=50, range=(0, 200), alpha=0.55,
             label="GNN $p_z$", color="#9467bd")
    ax2.axvline(M_W, ls="--", color="k", label=f"$m_W$={M_W:.0f}")
    ax2.set_xlabel("reconstructed $m(\\ell\\nu)$ [GeV]"); ax2.set_ylabel("events")
    ax2.set_title(f"No-solution events ({len(uns)}): analytic vs GNN"); ax2.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "neutrino_pz_gnn.png")
    os.makedirs(RESULTS_DIR, exist_ok=True); fig.savefig(out, dpi=130); plt.close(fig)
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "neutrino_pz_gnn.pt"))
    print(f"[gnn-reg] saved -> {out}")
    return {"rmse": rmse}


def main():
    cfg = _load_config(os.path.join(REPO_ROOT, "config.yaml"))
    root_dir = _abspath(cfg["data"]["root_dir"])
    sig, _ = _real_data_files(cfg, root_dir)
    if sig:
        print(f"[gnn-reg] using REAL ATLAS Open Data: {len(sig)} W->l nu files")
    else:
        sig = sorted(glob.glob(os.path.join(root_dir, "sample_signal*.root")))
        if not sig:
            raise FileNotFoundError("No W->l nu files. Run scripts/download_data.py or tests/make_sample.py")
        print("[gnn-reg] using SYNTHETIC fixtures")
    run(sig, cfg)


if __name__ == "__main__":
    main()
