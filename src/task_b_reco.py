"""Task B -- reconstruct the neutrino and the W boson from W -> l nu events.

Task A asked *whether* an event has a neutrino. Task B reconstructs it. The neutrino's
transverse momentum is the MET; its longitudinal momentum ``pz`` is unmeasured, so we
solve the W-mass constraint (``src/physics.neutrino_pz``) for the two candidate ``pz``
values, pick one, and rebuild the (lepton + neutrino) invariant mass.

We produce the headline Task-B figure: the reconstructed W-mass distribution alongside
the transverse mass ``mT``. Physics to expect:

* **mT** is bounded above by ``m_W`` with a Jacobian edge near 80 GeV (it only uses the
  transverse plane -- no ``pz``).
* the **reconstructed W mass** sits exactly at ``m_W`` for every event whose constraint
  has a real solution (by construction), with a tail below ``m_W`` for events where the
  MET is smeared enough that the discriminant goes negative and we keep the real part.
  The fraction in the peak is a direct measure of how often the analytic reconstruction
  has a physical solution.

Usage:
    python -m src.task_b_reco          # real W -> l nu data if present, else synthetic
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import evaluation, physics, read_root, selection
from src.pipeline import _abspath, _load_config, _real_data_files, REPO_ROOT

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
M_W = 80.379


def _lead(vec_array, mask):
    """Leading (index-0) element of each per-event vector, restricted to `mask`."""
    return np.array([float(np.asarray(v)[0]) for v in vec_array[mask]], dtype=float)


def run_reco(signal_files, cfg, out_png=None):
    """Reconstruct the W from the signal events and plot the mass resolution."""
    out_png = out_png or os.path.join(RESULTS_DIR, "wmass_reco.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    events = read_root.read_events(signal_files, tree="mini")
    mask, cutflow = selection.select(events, cfg)
    n_sel = int(np.asarray(mask).sum())
    print(f"[task-b] selected {n_sel} W -> l nu events")
    print(f"[task-b] cutflow: {cutflow}")
    if n_sel == 0:
        raise RuntimeError("No events pass selection -- nothing to reconstruct.")

    # Analytic reconstruction: neutrino pz from the W-mass constraint (min |pz| root).
    w_reco = evaluation.reconstructed_w_mass(events, mask, nu_pz_choice="min_abs")

    # Transverse mass for comparison (transverse plane only, no pz).
    lep_pt = _lead(events["lep_pt"], mask)
    lep_phi = _lead(events["lep_phi"], mask)
    met_et = np.asarray(events["met_et"], float)[mask]
    met_phi = np.asarray(events["met_phi"], float)[mask]
    mt = physics.transverse_mass(lep_pt, lep_phi, met_et, met_phi)

    # Fraction with a real (physical) neutrino-pz solution: those reconstruct to m_W.
    real_soln = np.abs(w_reco - M_W) < 0.5
    frac_real = float(real_soln.mean())

    print(f"[task-b] reconstructed W mass: median={np.median(w_reco):.2f} GeV, "
          f"mean={w_reco.mean():.2f} GeV")
    print(f"[task-b] fraction with a real pz solution (peak at m_W): {frac_real:.3f}")
    print(f"[task-b] transverse mass: median={np.median(mt):.2f} GeV "
          f"(Jacobian edge near m_W)")

    # --- figure: mT and reconstructed W mass ------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.hist(mt, bins=60, range=(0, 160), color="#1f77b4", alpha=0.85)
    ax1.axvline(M_W, ls="--", color="k", label=f"$m_W$ = {M_W:.1f} GeV")
    ax1.set_xlabel(r"transverse mass $m_T$ [GeV]")
    ax1.set_ylabel("events")
    ax1.set_title("Transverse mass (Jacobian edge)")
    ax1.legend()

    ax2.hist(w_reco, bins=60, range=(0, 160), color="#d62728", alpha=0.85)
    ax2.axvline(M_W, ls="--", color="k", label=f"$m_W$ = {M_W:.1f} GeV")
    ax2.set_xlabel(r"reconstructed $m(\ell\nu)$ [GeV]")
    ax2.set_ylabel("events")
    ax2.set_title(f"Reconstructed W mass ({frac_real*100:.0f}% real solutions)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[task-b] saved -> {out_png}")
    return {"w_reco": w_reco, "mt": mt, "frac_real": frac_real}


def main() -> None:
    cfg = _load_config(os.path.join(REPO_ROOT, "config.yaml"))
    root_dir = _abspath(cfg["data"]["root_dir"])

    signal_files, _ = _real_data_files(cfg, root_dir)
    if signal_files:
        print(f"[task-b] using REAL ATLAS Open Data: {len(signal_files)} W->l nu files")
    else:
        signal_files = sorted(glob.glob(os.path.join(root_dir, "sample_signal*.root")))
        if not signal_files:
            raise FileNotFoundError(
                "No W->l nu files. Run  python scripts/download_data.py  "
                "or  python tests/make_sample.py"
            )
        print(f"[task-b] using SYNTHETIC fixtures: {signal_files}")

    run_reco(signal_files, cfg)


if __name__ == "__main__":
    main()
