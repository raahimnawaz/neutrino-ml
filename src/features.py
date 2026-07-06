"""Feature engineering for the W -> l nu classifier (Task A).

Turns the per-event ATLAS Open Data record (as returned by ``read_root.read_events``)
into a flat ``pandas.DataFrame`` -- one row per *selected* event -- whose columns are
exactly the discriminating variables the classifier consumes.

All kinematic quantities are already in GeV by the time they reach this module (the
read layer converts ATLAS Open Data's native MeV -> GeV). We select exactly one lepton
per event (``lep_n == 1``), so every per-lepton quantity is taken at index ``[0]``.

The column set is fixed by the shared project contract::

    lep_pt, met_et, mt, dphi_lep_met, lep_eta, lep_iso_ptcone, jet_n, ht, lep_type

and must line up with ``config.yaml``'s ``features:`` list.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import physics

# Exact, ordered column contract. Kept as a module constant so downstream code
# (dataset assembly, model training) can import it instead of hard-coding names.
FEATURE_COLUMNS: list[str] = [
    "lep_pt",
    "met_et",
    "mt",
    "dphi_lep_met",
    "lep_eta",
    "lep_iso_ptcone",
    "jet_n",
    "ht",
    "lep_type",
]


def _lead(vec: Any) -> float:
    """Return the leading (index-0) element of a per-object vector as a float.

    ``read_events`` yields per-lepton / per-jet branches as object arrays of small
    numpy arrays (one array per event). After masking to ``lep_n == 1`` events, each
    such per-event entry has length 1, so index ``[0]`` is the single lepton.
    """
    return float(np.asarray(vec)[0])


def _sum(vec: Any) -> float:
    """Scalar sum of a per-object vector (e.g. sum of jet pT -> HT).

    Returns ``0.0`` for an empty vector (an event with ``jet_n == 0``).
    """
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.sum())


def build_features(
    events: Mapping[str, Any],
    mask: Sequence[bool] | np.ndarray,
    cfg: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the classifier feature table for the selected events.

    Parameters
    ----------
    events
        Per-event record from ``read_root.read_events``: a mapping from branch name
        to a numpy array (scalar branches) or object array of numpy arrays (per-lepton
        / per-jet vector branches). Kinematics are in GeV.
    mask
        Boolean array (length == number of events) marking events that pass
        ``selection.select``. Only these rows appear in the output.
    cfg
        Full config mapping. Accepted for signature/contract consistency; feature
        definitions here are fixed and do not depend on config values.

    Returns
    -------
    pandas.DataFrame
        One row per selected event, columns exactly ``FEATURE_COLUMNS`` in that order.
        Derived quantities:

        * ``mt``            -- transverse mass mT(lepton, MET), via ``physics.transverse_mass``.
        * ``dphi_lep_met``  -- ``abs(delta_phi(lep_phi, met_phi))`` in (0, pi].
        * ``lep_iso_ptcone``-- ``lep_ptcone30[0] / lep_pt[0]`` (relative track isolation).
        * ``ht``            -- scalar sum of jet pT.
    """
    del cfg  # feature definitions are fixed by contract; config is intentionally unused

    mask = np.asarray(mask, dtype=bool)

    # Select only the passing events for every branch we touch. Scalar branches slice
    # directly; vector branches are object arrays and also slice by the same mask.
    lep_pt_v = np.asarray(events["lep_pt"], dtype=object)[mask]
    lep_eta_v = np.asarray(events["lep_eta"], dtype=object)[mask]
    lep_phi_v = np.asarray(events["lep_phi"], dtype=object)[mask]
    lep_type_v = np.asarray(events["lep_type"], dtype=object)[mask]
    lep_ptcone30_v = np.asarray(events["lep_ptcone30"], dtype=object)[mask]
    jet_pt_v = np.asarray(events["jet_pt"], dtype=object)[mask]

    met_et = np.asarray(events["met_et"], dtype=float)[mask]
    met_phi = np.asarray(events["met_phi"], dtype=float)[mask]
    jet_n = np.asarray(events["jet_n"])[mask]

    n_sel = int(mask.sum())

    # Leading-lepton scalars (index [0]; lep_n == 1 guaranteed by selection).
    lep_pt = np.array([_lead(v) for v in lep_pt_v], dtype=float)
    lep_eta = np.array([_lead(v) for v in lep_eta_v], dtype=float)
    lep_phi = np.array([_lead(v) for v in lep_phi_v], dtype=float)
    lep_type = np.array([_lead(v) for v in lep_type_v], dtype=float)
    lep_ptcone30 = np.array([_lead(v) for v in lep_ptcone30_v], dtype=float)

    # HT = scalar sum of jet pT (0 for jet-less events).
    ht = np.array([_sum(v) for v in jet_pt_v], dtype=float)

    # Derived kinematics via the shared physics module (vectorized).
    mt = physics.transverse_mass(lep_pt, lep_phi, met_et, met_phi)
    dphi_lep_met = np.abs(physics.delta_phi(lep_phi, met_phi))

    # Relative track isolation. Guard against a zero-pT lepton (should not occur after
    # selection, which imposes lep_pt > lep_pt_min_gev, but keep the table finite).
    with np.errstate(divide="ignore", invalid="ignore"):
        lep_iso_ptcone = np.where(lep_pt > 0.0, lep_ptcone30 / lep_pt, 0.0)

    df = pd.DataFrame(
        {
            "lep_pt": lep_pt,
            "met_et": met_et,
            "mt": np.asarray(mt, dtype=float),
            "dphi_lep_met": np.asarray(dphi_lep_met, dtype=float),
            "lep_eta": lep_eta,
            "lep_iso_ptcone": np.asarray(lep_iso_ptcone, dtype=float),
            "jet_n": np.asarray(jet_n).astype(int),
            "ht": ht,
            "lep_type": lep_type.astype(int),
        }
    )

    # Enforce exact column order per contract and a clean 0..N-1 index.
    df = df[FEATURE_COLUMNS].reset_index(drop=True)
    assert len(df) == n_sel, "feature rows must equal number of selected events"
    return df
