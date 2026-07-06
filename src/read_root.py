"""Read ATLAS Open Data ``mini`` ntuples into plain numpy via ROOT.RDataFrame.

The ATLAS Open Data 13 TeV education ntuples store one TTree named ``mini`` whose
kinematic branches are in **MeV**. Every downstream module in this repo works in
**GeV**, so this layer is the single place where the MeV->GeV conversion happens:
``lep_pt``, ``lep_E``, ``jet_pt``, ``jet_E`` and ``met_et`` are divided by 1000
(the per-lepton / per-jet branches element-wise).

Weight-related branches (``mcWeight``, ``XSection``, ``SumWeights`` and the
``scaleFactor_*`` set) only exist on Monte-Carlo samples. Real data files simply
do not have them; we quietly omit any branch that is missing from the file and let
``src.weights.event_weights`` fall back to unit weights.

Per-lepton / per-jet vector branches (length ``lep_n`` / ``jet_n``) are returned as
1-D numpy object arrays whose elements are themselves numpy arrays, so downstream
code can index the single selected lepton as e.g. ``events["lep_pt"][i][0]``.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

import ROOT

# ---------------------------------------------------------------------------
# Branch bookkeeping
# ---------------------------------------------------------------------------

# Scalar (one value per event) branches.
_SCALAR_BRANCHES: list[str] = [
    "lep_n",
    "met_et",
    "met_phi",
    "jet_n",
    "mcWeight",
    "XSection",
    "SumWeights",
    "scaleFactor_PILEUP",
    "scaleFactor_ELE",
    "scaleFactor_MUON",
    "scaleFactor_LepTRIGGER",
]

# Per-lepton vector branches (length == lep_n).
_LEP_BRANCHES: list[str] = [
    "lep_pt",
    "lep_eta",
    "lep_phi",
    "lep_E",
    "lep_type",
    "lep_charge",
    "lep_isTightID",
    "lep_ptcone30",
    "lep_etcone20",
]

# Per-jet vector branches (length == jet_n).
_JET_BRANCHES: list[str] = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_E",
]

# The full contract-defined branch list, in a stable order.
BRANCHES: list[str] = _SCALAR_BRANCHES + _LEP_BRANCHES + _JET_BRANCHES

# Weight branches are MC-only; missing ones are dropped silently (real data).
_OPTIONAL_BRANCHES: set[str] = {
    "mcWeight",
    "XSection",
    "SumWeights",
    "scaleFactor_PILEUP",
    "scaleFactor_ELE",
    "scaleFactor_MUON",
    "scaleFactor_LepTRIGGER",
}

# Branches whose energy/momentum values must be converted MeV -> GeV. The isolation
# cones (ptcone30/etcone20) are momentum/energy sums in MeV too, so they must be scaled
# alongside lep_pt -- otherwise the relative isolation ptcone30/lep_pt is off by 1000x.
_GEV_SCALE_BRANCHES: set[str] = {
    "lep_pt", "lep_E", "jet_pt", "jet_E", "met_et", "lep_ptcone30", "lep_etcone20",
}

_MEV_PER_GEV: float = 1000.0


def _to_object_array_of_arrays(col: object) -> np.ndarray:
    """Normalise an RDataFrame vector column to a 1-D object array of np arrays.

    ``AsNumpy`` returns per-event vector branches as a numpy object array whose
    entries are ``ROOT.VecOps.RVec`` (or already numpy arrays). We copy each entry
    into a contiguous ``float64`` numpy array so downstream code never has to know
    about ROOT types and can index ``[0]`` cheaply.
    """
    out = np.empty(len(col), dtype=object)
    for i, entry in enumerate(col):
        out[i] = np.asarray(entry, dtype=np.float64)
    return out


def _scale_vector_branch(col: np.ndarray) -> np.ndarray:
    """Divide every element of every per-event vector by 1000 (MeV -> GeV)."""
    out = np.empty(len(col), dtype=object)
    for i, entry in enumerate(col):
        out[i] = np.asarray(entry, dtype=np.float64) / _MEV_PER_GEV
    return out


def read_events(
    files: Union[str, Sequence[str]],
    tree: str = "mini",
) -> dict[str, np.ndarray]:
    """Read the requested files into a ``{branch_name: array}`` dict (GeV units).

    Parameters
    ----------
    files:
        A single path or a sequence of paths to ATLAS Open Data ``.root`` files.
    tree:
        Name of the TTree to read (``"mini"`` for ATLAS Open Data).

    Returns
    -------
    dict[str, numpy.ndarray]
        Scalar branches map to 1-D numpy arrays (one value per event). Per-lepton
        and per-jet vector branches map to 1-D object arrays whose entries are
        numpy arrays. Energy/momentum branches (``lep_pt``, ``lep_E``, ``jet_pt``,
        ``jet_E``, ``met_et``) are already converted to GeV. Weight branches that
        are absent from the file (real data) are simply not present in the dict.
    """
    # RDataFrame accepts a single path (str) or a std::vector<string>-like list.
    if isinstance(files, str):
        file_list: list[str] = [files]
    else:
        file_list = list(files)

    df = ROOT.RDataFrame(tree, file_list)

    # Only request branches that actually exist in the file, so real-data samples
    # (which lack the MC weight branches) don't raise. RDataFrame exposes the
    # column names via GetColumnNames().
    available = {str(name) for name in df.GetColumnNames()}
    requested: list[str] = []
    for name in BRANCHES:
        if name in available:
            requested.append(name)
        elif name not in _OPTIONAL_BRANCHES:
            # A required (non-weight) branch is genuinely missing: fail loudly.
            raise KeyError(
                f"Required branch {name!r} not found in tree {tree!r}. "
                f"Available columns: {sorted(available)}"
            )
        # Optional/weight branch missing -> silently skip (handled by weights.py).

    raw = df.AsNumpy(requested)

    events: dict[str, np.ndarray] = {}
    lep_set = set(_LEP_BRANCHES)
    jet_set = set(_JET_BRANCHES)

    for name, col in raw.items():
        if name in lep_set or name in jet_set:
            # Per-event vector branch.
            if name in _GEV_SCALE_BRANCHES:
                events[name] = _scale_vector_branch(col)
            else:
                events[name] = _to_object_array_of_arrays(col)
        else:
            # Scalar branch (one value per event).
            arr = np.asarray(col)
            if name in _GEV_SCALE_BRANCHES:
                arr = arr.astype(np.float64) / _MEV_PER_GEV
            events[name] = arr

    return events
