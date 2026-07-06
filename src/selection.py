"""Sequential W->l nu event selection with a cutflow.

The signal topology is a single isolated high-pT lepton recoiling against a
neutrino, whose only signature is missing transverse energy (MET). The cuts below
are applied *sequentially* (each on the survivors of the previous one) so the
returned cutflow is a monotonically decreasing list of ``(cut_name, n_surviving)``
counts — the standard way an ATLAS analysis reports its selection.

All per-lepton quantities use index ``[0]`` because the ``lep_n == 1`` cut runs
first; every later per-lepton cut is only ever *evaluated* on events that already
have exactly one lepton, so ``[0]`` is always safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.physics import transverse_mass


def _lep0_scalar(events: dict, branch: str, active: np.ndarray) -> np.ndarray:
    """Extract the index-0 element of a per-lepton vector branch.

    The value is only read where ``active`` is True (events that survived the
    ``lep_n == 1`` cut and therefore definitely have a lepton). Inactive slots get
    ``0.0`` as a harmless placeholder; they are already excluded by the mask.
    """
    col = events[branch]
    out = np.zeros(len(col), dtype=np.float64)
    for i in np.flatnonzero(active):
        out[i] = float(np.asarray(col[i], dtype=np.float64)[0])
    return out


def select(events: dict, cfg: dict) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Apply the sequential W->l nu selection.

    Parameters
    ----------
    events:
        ``{branch: array}`` dict from :func:`src.read_root.read_events` (GeV units).
    cfg:
        Parsed ``config.yaml``; the ``cfg["selection"]`` block supplies the cuts:
        ``lep_n``, ``require_tight_id``, ``lep_pt_min_gev``, ``iso_ptcone_frac_max``,
        ``iso_etcone_frac_max``, ``met_min_gev`` and (optionally) ``mt_min_gev``.

    Returns
    -------
    (mask, cutflow):
        ``mask`` is a boolean numpy array (True == event passes every cut).
        ``cutflow`` is a list of ``(cut_name, surviving_count)`` tuples including a
        leading ``("all", n_events)`` entry, in application order.
    """
    sel: dict[str, Any] = cfg["selection"]

    lep_n = np.asarray(events["lep_n"]).astype(np.int64)
    met_et = np.asarray(events["met_et"], dtype=np.float64)
    met_phi = np.asarray(events["met_phi"], dtype=np.float64)
    n_events = len(lep_n)

    cutflow: list[tuple[str, int]] = [("all", int(n_events))]

    # --- Cut 1: exactly one lepton -----------------------------------------
    required_lep_n = int(sel.get("lep_n", 1))
    mask = lep_n == required_lep_n
    cutflow.append((f"lep_n=={required_lep_n}", int(mask.sum())))

    # From here on, every event still in `mask` has exactly one lepton, so the
    # index-0 accessors below are always valid.

    # --- Cut 2: tight lepton ID (optional) ---------------------------------
    if sel.get("require_tight_id", False):
        is_tight = _lep0_scalar(events, "lep_isTightID", mask)
        mask = mask & (is_tight != 0.0)
        cutflow.append(("lep_isTightID", int(mask.sum())))

    # --- Cut 3: leading-lepton pT ------------------------------------------
    lep_pt0 = _lep0_scalar(events, "lep_pt", mask)
    mask = mask & (lep_pt0 > float(sel["lep_pt_min_gev"]))
    cutflow.append((f"lep_pt>{sel['lep_pt_min_gev']}", int(mask.sum())))

    # --- Cut 4: track isolation (ptcone30 / pt) ----------------------------
    ptcone30 = _lep0_scalar(events, "lep_ptcone30", mask)
    # lep_pt0 is >0 wherever mask is True after cut 3; avoid divide-by-zero.
    safe_pt = np.where(lep_pt0 > 0.0, lep_pt0, 1.0)
    iso_ptcone = ptcone30 / safe_pt
    mask = mask & (iso_ptcone < float(sel["iso_ptcone_frac_max"]))
    cutflow.append(("iso_ptcone", int(mask.sum())))

    # --- Cut 5: calorimeter isolation (etcone20 / pt) ----------------------
    etcone20 = _lep0_scalar(events, "lep_etcone20", mask)
    iso_etcone = etcone20 / safe_pt
    mask = mask & (iso_etcone < float(sel["iso_etcone_frac_max"]))
    cutflow.append(("iso_etcone", int(mask.sum())))

    # --- Cut 6: missing transverse energy ----------------------------------
    mask = mask & (met_et > float(sel["met_min_gev"]))
    cutflow.append((f"met>{sel['met_min_gev']}", int(mask.sum())))

    # --- Cut 7: transverse mass floor (optional) ---------------------------
    mt_min = float(sel.get("mt_min_gev", 0.0))
    if mt_min > 0.0:
        # transverse_mass is vectorised; feed it leading-lepton pt/phi + MET.
        lep_phi0 = _lep0_scalar(events, "lep_phi", mask)
        mt = transverse_mass(lep_pt0, lep_phi0, met_et, met_phi)
        mask = mask & (np.asarray(mt, dtype=np.float64) > mt_min)
        cutflow.append((f"mt>{mt_min}", int(mask.sum())))

    return mask, cutflow
