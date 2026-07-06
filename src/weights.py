"""Per-event Monte-Carlo weights for ATLAS Open Data.

A physics analysis compares data to Monte-Carlo simulation, and every simulated
event carries a weight so that the sum of weights reproduces the number of events
expected for a given integrated luminosity. The ATLAS Open Data convention is::

    w = mcWeight
        * scaleFactor_PILEUP * scaleFactor_ELE
        * scaleFactor_MUON   * scaleFactor_LepTRIGGER
        * (XSection * lumi_pb / SumWeights)

where ``XSection`` is the process cross-section (pb), ``lumi_pb`` the integrated
luminosity (pb^-1) and ``SumWeights`` the sum of generator weights of the full
sample (normalisation denominator). The ``scaleFactor_*`` terms are per-event
data/MC correction factors.

Real *data* files carry none of these branches, so this function returns unit
weights (``np.ones``) whenever any required branch is missing.
"""

from __future__ import annotations

import numpy as np

# The exact set of branches needed to build an MC weight.
_MC_SCALAR_WEIGHT_BRANCHES: tuple[str, ...] = (
    "mcWeight",
    "scaleFactor_PILEUP",
    "scaleFactor_ELE",
    "scaleFactor_MUON",
    "scaleFactor_LepTRIGGER",
    "XSection",
    "SumWeights",
)


def _n_events(events: dict) -> int:
    """Infer the event count from any present branch (``lep_n`` is always there)."""
    for key in ("lep_n", "met_et", "met_phi", "jet_n"):
        if key in events:
            return len(events[key])
    # Fall back to whatever branch exists.
    for value in events.values():
        return len(value)
    return 0


def event_weights(events: dict, lumi_pb: float) -> np.ndarray:
    """Return the per-event MC weight array (GeV-agnostic; weights are unitless).

    Parameters
    ----------
    events:
        The ``{branch: array}`` dict produced by :func:`src.read_root.read_events`.
    lumi_pb:
        Integrated luminosity in pb^-1 (e.g. ``config["data"]["lumi_pb"]``).

    Returns
    -------
    numpy.ndarray
        A 1-D ``float64`` array of length ``n_events``. If any weight branch is
        absent (real data), an array of ones is returned instead.
    """
    n = _n_events(events)

    # If this is real data (or any weight branch is missing), use unit weights.
    if any(branch not in events for branch in _MC_SCALAR_WEIGHT_BRANCHES):
        return np.ones(n, dtype=np.float64)

    mc_weight = np.asarray(events["mcWeight"], dtype=np.float64)
    sf_pileup = np.asarray(events["scaleFactor_PILEUP"], dtype=np.float64)
    sf_ele = np.asarray(events["scaleFactor_ELE"], dtype=np.float64)
    sf_muon = np.asarray(events["scaleFactor_MUON"], dtype=np.float64)
    sf_trig = np.asarray(events["scaleFactor_LepTRIGGER"], dtype=np.float64)
    xsection = np.asarray(events["XSection"], dtype=np.float64)
    sum_weights = np.asarray(events["SumWeights"], dtype=np.float64)

    # Normalisation to the requested luminosity. SumWeights is the sample-wide
    # generator-weight sum; guard against a zero denominator just in case.
    lumi_norm = np.divide(
        xsection * float(lumi_pb),
        sum_weights,
        out=np.zeros_like(xsection, dtype=np.float64),
        where=sum_weights != 0,
    )

    weights = (
        mc_weight
        * sf_pileup
        * sf_ele
        * sf_muon
        * sf_trig
        * lumi_norm
    )
    return weights.astype(np.float64)
