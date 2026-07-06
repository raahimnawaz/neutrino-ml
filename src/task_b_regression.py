"""Task B (stretch) -- ML regression of the neutrino longitudinal momentum ``pz``.

The analytic W-mass reconstruction (``src/physics.neutrino_pz``) has NO real solution
for ~40% of real W -> l nu events: the MET is smeared enough that the discriminant goes
negative, and we can only fall back to the real part (which biases the reconstructed W
mass high). This module trains a regressor to predict ``pz`` from the event kinematics so
we have a physical estimate for *every* event, including the ones the analytic method
cannot solve.

The reduced ATLAS Open Data has no truth-level neutrino ``pz`` to supervise against
(verified: only reco branches + ``*_truthMatched`` flags exist). So we supervise on the
analytic solution where it exists:

    * TARGET  = analytic pz (the smaller-|pz| root) on events WITH a real solution.
    * TRAIN   on a subset of those solvable events.
    * VALIDATE on held-out solvable events -> a genuine metric (does the ML reproduce
      the physics-correct pz?).
    * APPLY   to the unsolvable events -> a pz estimate the analytic method could not give.

Usage:
    python -m src.task_b_regression
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src import physics, read_root, selection
from src.pipeline import _abspath, _load_config, _real_data_files, REPO_ROOT

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "saved_models")
M_W = 80.379

# Observable features the regressor sees (the raw ingredients of the W-mass quadratic
# plus a little event context). Deliberately the same information the analytic solver
# uses, so the model can learn the mapping and extrapolate past the no-solution boundary.
FEATURES = [
    "lep_px", "lep_py", "lep_pz", "lep_E",
    "met_x", "met_y", "met_et", "mt",
    "lep_pt", "dphi_lep_met", "jet_n", "lep_type",
]


def _lead(vec_array, mask):
    return np.array([float(np.asarray(v)[0]) for v in vec_array[mask]], dtype=float)


def build_table(events, mask) -> pd.DataFrame:
    """Per selected event: observable features + the analytic pz target + has_real flag."""
    mask = np.asarray(mask, dtype=bool)
    lep_pt = _lead(events["lep_pt"], mask)
    lep_eta = _lead(events["lep_eta"], mask)
    lep_phi = _lead(events["lep_phi"], mask)
    lep_E = _lead(events["lep_E"], mask)
    lep_type = _lead(events["lep_type"], mask)
    met_et = np.asarray(events["met_et"], float)[mask]
    met_phi = np.asarray(events["met_phi"], float)[mask]
    jet_n = np.asarray(events["jet_n"])[mask].astype(float)

    lep_px = lep_pt * np.cos(lep_phi)
    lep_py = lep_pt * np.sin(lep_phi)
    lep_pz = lep_pt * np.sinh(lep_eta)
    met_x = met_et * np.cos(met_phi)
    met_y = met_et * np.sin(met_phi)
    mt = physics.transverse_mass(lep_pt, lep_phi, met_et, met_phi)
    dphi = np.abs(physics.delta_phi(lep_phi, met_phi))

    n = lep_pt.shape[0]
    analytic_pz = np.empty(n)
    has_real = np.empty(n, dtype=bool)
    for i in range(n):
        pz1, pz2 = physics.neutrino_pz(lep_px[i], lep_py[i], lep_pz[i], lep_E[i],
                                       met_x[i], met_y[i], mW=M_W)
        # smaller-|pz| root (the standard central-neutrino choice)
        analytic_pz[i] = pz1 if abs(pz1) <= abs(pz2) else pz2
        # a real (two-solution) event has distinct roots; the disc<0 fallback returns
        # equal roots and reconstructs to a mass above m_W.
        m = physics.reconstruct_w_mass(lep_px[i], lep_py[i], lep_pz[i], lep_E[i],
                                       met_x[i], met_y[i], analytic_pz[i])
        has_real[i] = abs(m - M_W) < 0.5

    return pd.DataFrame({
        "lep_px": lep_px, "lep_py": lep_py, "lep_pz": lep_pz, "lep_E": lep_E,
        "met_x": met_x, "met_y": met_y, "met_et": met_et, "mt": mt,
        "lep_pt": lep_pt, "dphi_lep_met": dphi, "jet_n": jet_n, "lep_type": lep_type,
        "analytic_pz": analytic_pz, "has_real": has_real,
    })


def run(signal_files, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    events = read_root.read_events(signal_files, tree="mini")
    mask, cutflow = selection.select(events, cfg)
    print(f"[reg] selected {int(np.asarray(mask).sum())} W -> l nu events")

    tbl = build_table(events, mask)
    solvable = tbl[tbl["has_real"]].reset_index(drop=True)
    unsolvable = tbl[~tbl["has_real"]].reset_index(drop=True)
    print(f"[reg] analytic real solutions: {len(solvable)} "
          f"({100*len(solvable)/len(tbl):.1f}%)  |  no solution: {len(unsolvable)}")

    # --- train on solvable events, validate on a held-out slice ----------------------
    rs = int(cfg["train"]["random_state"])
    tr, te = train_test_split(solvable, test_size=0.3, random_state=rs)
    reg = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                       n_jobs=-1, random_state=rs)
    reg.fit(tr[FEATURES], tr["analytic_pz"])

    pred_te = reg.predict(te[FEATURES])
    rmse = float(np.sqrt(np.mean((pred_te - te["analytic_pz"]) ** 2)))
    # relative to the spread of the target, so the number is interpretable
    frac = rmse / float(te["analytic_pz"].std())
    print(f"[reg] held-out validation (vs analytic pz): "
          f"RMSE={rmse:.1f} GeV  ({frac:.2f} x target std)")

    # --- apply to the events the analytic method could NOT solve ---------------------
    # analytic fallback there = the real part (what neutrino_pz returns) -> biased high.
    ml_pz_uns = reg.predict(unsolvable[FEATURES])
    m_analytic = physics.reconstruct_w_mass(
        unsolvable["lep_px"], unsolvable["lep_py"], unsolvable["lep_pz"],
        unsolvable["lep_E"], unsolvable["met_x"], unsolvable["met_y"],
        unsolvable["analytic_pz"].to_numpy())
    m_ml = physics.reconstruct_w_mass(
        unsolvable["lep_px"], unsolvable["lep_py"], unsolvable["lep_pz"],
        unsolvable["lep_E"], unsolvable["met_x"], unsolvable["met_y"], ml_pz_uns)
    print(f"[reg] reconstructed W mass on no-solution events: "
          f"analytic-fallback median={np.median(m_analytic):.1f} GeV  ->  "
          f"ML median={np.median(m_ml):.1f} GeV  (target {M_W:.1f})")

    # --- figure ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.scatter(te["analytic_pz"], pred_te, s=3, alpha=0.2, color="#1f77b4")
    lim = np.percentile(np.abs(te["analytic_pz"]), 99)
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("analytic $p_z^\\nu$ [GeV]"); ax1.set_ylabel("ML $p_z^\\nu$ [GeV]")
    ax1.set_title(f"Validation on solvable events\nRMSE = {rmse:.1f} GeV")

    ax2.hist(np.clip(m_analytic, 0, 200), bins=50, range=(0, 200), alpha=0.6,
             label="analytic fallback", color="#d62728")
    ax2.hist(np.clip(m_ml, 0, 200), bins=50, range=(0, 200), alpha=0.6,
             label="ML $p_z$", color="#2ca02c")
    ax2.axvline(M_W, ls="--", color="k", label=f"$m_W$={M_W:.0f}")
    ax2.set_xlabel("reconstructed $m(\\ell\\nu)$ [GeV]"); ax2.set_ylabel("events")
    ax2.set_title(f"No-solution events ({len(unsolvable)}): analytic vs ML")
    ax2.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "neutrino_pz_regression.png")
    fig.savefig(out, dpi=130); plt.close(fig)

    reg.save_model(os.path.join(MODELS_DIR, "neutrino_pz_regressor.json"))
    print(f"[reg] saved -> {out}")
    print(f"[reg] saved model -> {os.path.join(MODELS_DIR, 'neutrino_pz_regressor.json')}")
    return {"rmse": rmse, "n_solvable": len(solvable), "n_unsolvable": len(unsolvable)}


def main():
    cfg = _load_config(os.path.join(REPO_ROOT, "config.yaml"))
    root_dir = _abspath(cfg["data"]["root_dir"])
    signal_files, _ = _real_data_files(cfg, root_dir)
    if signal_files:
        print(f"[reg] using REAL ATLAS Open Data: {len(signal_files)} W->l nu files")
    else:
        signal_files = sorted(glob.glob(os.path.join(root_dir, "sample_signal*.root")))
        if not signal_files:
            raise FileNotFoundError("No W->l nu files. Run scripts/download_data.py or "
                                    "tests/make_sample.py")
        print(f"[reg] using SYNTHETIC fixtures: {signal_files}")
    run(signal_files, cfg)


if __name__ == "__main__":
    main()
