"""Evaluation utilities for the neutrino-ml analysis.

This module scores a trained classifier the way a real ATLAS analysis does:
not by plain accuracy but by *weighted* figures of merit (weighted ROC AUC,
expected discovery significance, and the signal-improvement characteristic),
where every event carries its physics ``weight`` (cross-section x luminosity /
sum-of-weights, times MC scale factors). It also provides the neutrino-pz
reconstruction sanity check (Task B): the invariant mass of the reconstructed
(lepton + neutrino) system should pile up near the W mass.

All momenta/energies are in GeV (the read layer already converted from MeV).

Public API (see the project CONTRACT):
    weighted_auc(y, scores, w) -> float
    significance_scan(y, scores, w) -> dict
    sic_curve(y, scores, w) -> (signal_eff, sic)
    plot_roc(y, scores, w, out) -> str
    plot_significance(y, scores, w, out) -> str
    reconstructed_w_mass(events, mask, nu_pz_choice="min_abs") -> np.ndarray
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

from . import physics

__all__ = [
    "weighted_auc",
    "significance_scan",
    "sic_curve",
    "plot_roc",
    "plot_significance",
    "reconstructed_w_mass",
]


# --------------------------------------------------------------------------- #
# Weighted classifier metrics
# --------------------------------------------------------------------------- #
def weighted_auc(y: np.ndarray, scores: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC AUC.

    Thin wrapper over :func:`sklearn.metrics.roc_auc_score` with
    ``sample_weight=w`` so each event contributes its physics weight.

    Parameters
    ----------
    y : array of {0, 1}
        True labels (1 = signal, 0 = background).
    scores : array of float
        Classifier scores / probabilities for the positive (signal) class.
    w : array of float
        Per-event weights.

    Returns
    -------
    float
        The weighted area under the ROC curve.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    # ROC is a ranking metric and sklearn requires non-negative weights; real NLO MC
    # has negative-weight events, so use |w| here. (Signed weights are kept for the
    # significance yields, where a negative event physically subtracts.)
    w = np.abs(np.asarray(w, dtype=float))
    return float(roc_auc_score(y, scores, sample_weight=w))


def significance_scan(y: np.ndarray, scores: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """Scan a cut on the classifier score for the best expected significance.

    For every candidate threshold we count the *weighted* signal (S) and
    background (B) yields that pass ``score >= threshold`` and evaluate two
    common figures of merit:

    * the simple ``Z = S / sqrt(B)`` (valid in the large-B limit), and
    * the Asimov significance
      ``Z_A = sqrt( 2 * ((S + B) * ln(1 + S/B) - S) )``,
      which is the median expected discovery significance of a counting
      experiment (Cowan et al., Eur. Phys. J. C 71 (2011) 1554) and reduces
      to ``S/sqrt(B)`` when ``S << B``.

    The scan reports the threshold maximising each of the two, so a caller can
    compare the naive and the correct estimator.

    Parameters
    ----------
    y : array of {0, 1}
        True labels (1 = signal, 0 = background).
    scores : array of float
        Classifier scores for the signal class.
    w : array of float
        Per-event weights (signal and background yields both come from these).

    Returns
    -------
    dict
        Keys:
        ``best_threshold``       -- score cut maximising ``S/sqrt(B)``;
        ``best_significance``    -- that maximal ``S/sqrt(B)``;
        ``best_threshold_asimov``-- score cut maximising the Asimov ``Z_A``;
        ``best_asimov``          -- that maximal Asimov significance;
        ``s_at_best``            -- weighted signal yield at ``best_threshold``;
        ``b_at_best``            -- weighted background yield at ``best_threshold``.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    w = np.asarray(w, dtype=float)

    is_sig = y == 1
    is_bkg = ~is_sig

    # Candidate thresholds: every distinct score is a place the yields change.
    # Sorting descending lets us accumulate "passing" yields with a cumulative
    # sum, so the whole scan is O(n log n) instead of O(n^2).
    order = np.argsort(scores)[::-1]
    s_order = scores[order]
    w_sig_order = np.where(is_sig[order], w[order], 0.0)
    w_bkg_order = np.where(is_bkg[order], w[order], 0.0)

    # Cumulative weighted yields for score >= s_order[i].
    s_cum = np.cumsum(w_sig_order)
    b_cum = np.cumsum(w_bkg_order)

    # Collapse ties: only keep the last index of each run of equal scores, so a
    # threshold corresponds to "all events with this score or higher pass".
    keep = np.ones(s_order.shape[0], dtype=bool)
    keep[:-1] = s_order[1:] != s_order[:-1]

    thr = s_order[keep]
    s_pass = s_cum[keep]
    b_pass = b_cum[keep]

    # --- Z = S / sqrt(B) (guard B <= 0) ---------------------------------- #
    z_simple = np.zeros_like(s_pass)
    good = b_pass > 0.0
    z_simple[good] = s_pass[good] / np.sqrt(b_pass[good])

    # --- Asimov Z_A (guard B <= 0 and S <= 0) ---------------------------- #
    z_asimov = _asimov_significance(s_pass, b_pass)

    if thr.size == 0 or not np.any(good):
        # No usable threshold (e.g. no background events at all).
        return {
            "best_threshold": float("nan"),
            "best_significance": 0.0,
            "best_threshold_asimov": float("nan"),
            "best_asimov": 0.0,
            "s_at_best": 0.0,
            "b_at_best": 0.0,
        }

    i_simple = int(np.argmax(z_simple))
    i_asimov = int(np.argmax(z_asimov))

    return {
        "best_threshold": float(thr[i_simple]),
        "best_significance": float(z_simple[i_simple]),
        "best_threshold_asimov": float(thr[i_asimov]),
        "best_asimov": float(z_asimov[i_asimov]),
        "s_at_best": float(s_pass[i_simple]),
        "b_at_best": float(b_pass[i_simple]),
    }


def _asimov_significance(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised Asimov significance ``sqrt(2*((S+B)*ln(1+S/B) - S))``.

    Returns 0 wherever it is undefined (``B <= 0`` or ``S <= 0``) rather than
    nan, so the surrounding argmax is well behaved.
    """
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros(np.broadcast(s, b).shape, dtype=float)
    valid = (b > 0.0) & (s > 0.0)
    sv = s[valid]
    bv = b[valid]
    radicand = 2.0 * ((sv + bv) * np.log1p(sv / bv) - sv)
    # Numerical noise can push a tiny radicand slightly negative; clip at 0.
    out[valid] = np.sqrt(np.clip(radicand, 0.0, None))
    return out


def sic_curve(
    y: np.ndarray, scores: np.ndarray, w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Signal-improvement characteristic (SIC).

    The SIC is ``tpr / sqrt(fpr)`` as a function of the signal efficiency
    (``tpr``); it measures how much a cut on the classifier improves
    ``S/sqrt(B)`` relative to no cut, and its maximum is the best achievable
    significance-gain factor.

    Parameters
    ----------
    y : array of {0, 1}
        True labels (1 = signal, 0 = background).
    scores : array of float
        Classifier scores for the signal class.
    w : array of float
        Per-event weights (weighted ``tpr`` / ``fpr``).

    Returns
    -------
    (signal_eff, sic) : tuple of np.ndarray
        ``signal_eff`` is the weighted true-positive rate (signal efficiency);
        ``sic`` is ``tpr / sqrt(fpr)`` at the same thresholds. Points where
        ``fpr == 0`` (SIC undefined / infinite) are dropped from both arrays.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    w = np.abs(np.asarray(w, dtype=float))  # ranking metric; |w| for negative NLO weights

    fpr, tpr, _ = roc_curve(y, scores, sample_weight=w)

    # tpr/sqrt(fpr) is undefined at fpr == 0 (the top-left corner); drop those.
    good = fpr > 0.0
    signal_eff = tpr[good]
    sic = tpr[good] / np.sqrt(fpr[good])
    return signal_eff, sic


# --------------------------------------------------------------------------- #
# Plots (saved under results/)
# --------------------------------------------------------------------------- #
def plot_roc(y: np.ndarray, scores: np.ndarray, w: np.ndarray, out: str) -> str:
    """Draw the weighted ROC curve and save it to ``out`` (a PNG path).

    Returns the path written (so callers can log it).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed on WSL/CI.
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    w = np.abs(np.asarray(w, dtype=float))  # ranking metric; |w| for negative NLO weights

    fpr, tpr, _ = roc_curve(y, scores, sample_weight=w)
    auc = weighted_auc(y, scores, w)

    _ensure_parent(out)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(fpr, tpr, lw=2.0, label=f"classifier (AUC = {auc:.3f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], ls="--", color="gray", lw=1.0, label="random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate (background eff.)")
    ax.set_ylabel("True positive rate (signal eff.)")
    ax.set_title("Weighted ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_significance(y: np.ndarray, scores: np.ndarray, w: np.ndarray, out: str) -> str:
    """Draw expected significance vs. score threshold and save to ``out``.

    Both ``S/sqrt(B)`` and the Asimov ``Z_A`` are shown, with the best Asimov
    working point marked. Returns the path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    w = np.asarray(w, dtype=float)

    is_sig = y == 1
    is_bkg = ~is_sig

    order = np.argsort(scores)[::-1]
    s_order = scores[order]
    w_sig_order = np.where(is_sig[order], w[order], 0.0)
    w_bkg_order = np.where(is_bkg[order], w[order], 0.0)

    s_cum = np.cumsum(w_sig_order)
    b_cum = np.cumsum(w_bkg_order)

    keep = np.ones(s_order.shape[0], dtype=bool)
    keep[:-1] = s_order[1:] != s_order[:-1]

    thr = s_order[keep]
    s_pass = s_cum[keep]
    b_pass = b_cum[keep]

    z_simple = np.zeros_like(s_pass)
    good = b_pass > 0.0
    z_simple[good] = s_pass[good] / np.sqrt(b_pass[good])
    z_asimov = _asimov_significance(s_pass, b_pass)

    _ensure_parent(out)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(thr, z_simple, lw=2.0, label=r"$S/\sqrt{B}$")
    ax.plot(thr, z_asimov, lw=2.0, label=r"Asimov $Z_A$")

    if thr.size and np.any(z_asimov > 0):
        i_best = int(np.argmax(z_asimov))
        ax.axvline(thr[i_best], color="gray", ls="--", lw=1.0)
        ax.plot(
            [thr[i_best]],
            [z_asimov[i_best]],
            "o",
            color="crimson",
            label=f"best $Z_A$ = {z_asimov[i_best]:.2f} @ {thr[i_best]:.3f}",
        )

    ax.set_xlabel("Score threshold (cut: score $\\geq$ threshold)")
    ax.set_ylabel("Expected significance")
    ax.set_title("Significance scan")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Task B: neutrino-pz reconstruction sanity check
# --------------------------------------------------------------------------- #
def reconstructed_w_mass(
    events: Dict[str, np.ndarray],
    mask: np.ndarray,
    nu_pz_choice: str = "min_abs",
) -> np.ndarray:
    """Reconstruct the (lepton + neutrino) invariant mass for selected events.

    This is the Task-B sanity check: the neutrino longitudinal momentum ``pz``
    is not measured, so we solve the W-mass constraint (via
    :func:`physics.neutrino_pz`) for the two candidate solutions, pick one per
    the ``nu_pz_choice`` policy, and recompute the (lepton + neutrino)
    invariant mass with :func:`physics.reconstruct_w_mass`. When the input is
    self-consistent the resulting distribution peaks at the W mass.

    The lepton four-vector is taken from the single selected lepton (index 0):
    ``lep_pt, lep_eta, lep_phi, lep_E`` in GeV. The neutrino transverse
    momentum is the missing transverse energy: ``met_x = met_et*cos(met_phi)``,
    ``met_y = met_et*sin(met_phi)``.

    Parameters
    ----------
    events : dict[str, np.ndarray]
        The read-layer event dict (GeV). Uses ``lep_pt, lep_eta, lep_phi,
        lep_E`` (object arrays of per-lepton vectors, index 0) and ``met_et,
        met_phi`` (scalar per event).
    mask : np.ndarray[bool]
        Event-level selection mask (from :func:`selection.select`); the mass is
        computed only for events where ``mask`` is True.
    nu_pz_choice : {"min_abs", "max_abs", "pz1", "pz2"}, optional
        Which of the two quadratic solutions to keep:
        ``"min_abs"`` (default) the one with the smaller ``|pz|`` (the usual
        physics choice, closest to a central neutrino); ``"max_abs"`` the
        larger; ``"pz1"``/``"pz2"`` the first/second root as returned by the
        solver.

    Returns
    -------
    np.ndarray
        One reconstructed W mass (GeV) per selected event, in the order the
        events appear after applying ``mask``.
    """
    valid_choices = {"min_abs", "max_abs", "pz1", "pz2"}
    if nu_pz_choice not in valid_choices:
        raise ValueError(
            f"nu_pz_choice must be one of {sorted(valid_choices)}, got {nu_pz_choice!r}"
        )

    mask = np.asarray(mask, dtype=bool)

    # Per-lepton vectors are object arrays of small np arrays; take index 0.
    lep_pt = _first_component(events["lep_pt"])[mask]
    lep_eta = _first_component(events["lep_eta"])[mask]
    lep_phi = _first_component(events["lep_phi"])[mask]
    lep_E = _first_component(events["lep_E"])[mask]

    met_et = np.asarray(events["met_et"], dtype=float)[mask]
    met_phi = np.asarray(events["met_phi"], dtype=float)[mask]

    # Lepton Cartesian momentum from (pt, eta, phi).
    lep_px = lep_pt * np.cos(lep_phi)
    lep_py = lep_pt * np.sin(lep_phi)
    lep_pz = lep_pt * np.sinh(lep_eta)

    # Neutrino transverse momentum == MET.
    met_x = met_et * np.cos(met_phi)
    met_y = met_et * np.sin(met_phi)

    n = lep_pt.shape[0]
    masses = np.empty(n, dtype=float)
    for i in range(n):
        pz1, pz2 = physics.neutrino_pz(
            float(lep_px[i]),
            float(lep_py[i]),
            float(lep_pz[i]),
            float(lep_E[i]),
            float(met_x[i]),
            float(met_y[i]),
        )
        nu_pz = _choose_pz(pz1, pz2, nu_pz_choice)
        masses[i] = physics.reconstruct_w_mass(
            float(lep_px[i]),
            float(lep_py[i]),
            float(lep_pz[i]),
            float(lep_E[i]),
            float(met_x[i]),
            float(met_y[i]),
            nu_pz,
        )
    return masses


def _choose_pz(pz1: float, pz2: float, choice: str) -> float:
    """Select one neutrino-pz root per the requested policy."""
    if choice == "pz1":
        return pz1
    if choice == "pz2":
        return pz2
    if choice == "max_abs":
        return pz1 if abs(pz1) >= abs(pz2) else pz2
    # default: "min_abs"
    return pz1 if abs(pz1) <= abs(pz2) else pz2


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _first_component(vec_branch) -> np.ndarray:
    """Return element [0] of each per-lepton vector as a flat float array.

    The read layer stores per-lepton branches as an object array of small
    arrays (one per event). Since we select exactly one lepton (lep_n == 1),
    the physics quantity is always index 0. Works whether the branch is a numpy
    object array, an awkward array, or already a flat 1-D numeric array.
    """
    arr = np.asarray(vec_branch, dtype=object)
    # Already a flat numeric 1-D array (dtype!=object): treat as the [0] values.
    if arr.dtype != object:
        return np.asarray(vec_branch, dtype=float)
    return np.array([float(np.asarray(v)[0]) for v in arr], dtype=float)


def _ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` if it does not yet exist."""
    import os

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
