"""Write synthetic ATLAS-Open-Data-style fixtures so the whole pipeline is runnable
with NO real data.

Two files are produced under ``data/root/``:

* ``sample_signal.root`` -- W -> l nu -like events: exactly one isolated, high-pT
  lepton recoiling against genuine missing transverse energy (the invisible
  neutrino).  The lepton and the MET are arranged roughly back-to-back so that the
  transverse mass ``mT`` shows the characteristic Jacobian edge near the W mass.
* ``sample_bkg.root`` -- QCD / Z-ish background: also one lepton, but the MET is
  small (fake / mismeasured) and/or the lepton is poorly isolated, so ``mT`` is low.

Both TTrees are named ``mini`` and store **MeV** for all energy/momentum branches,
exactly like the real ATLAS Open Data ntuples.  The read layer (``src/read_root.py``)
is responsible for the MeV -> GeV conversion; here we multiply GeV quantities by 1000
before filling.

The file is written with :mod:`uproot`, so no PyROOT / ROOT install is required to
*generate* the fixtures (only to *read* them via RDataFrame later).

Run directly::

    python tests/make_sample.py
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import uproot

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO_ROOT, "data", "root")

N_SIGNAL = 3000
N_BKG = 3000
SEED = 1234

MEV_PER_GEV = 1000.0
LEP_MASS_GEV = 0.000511  # electron-ish; negligible vs. pT but keeps E >= |p|

# ATLAS Open Data lepton type codes
TYPE_ELECTRON = 11
TYPE_MUON = 13


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _lepton_four_vector(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray,
                        mass: float = LEP_MASS_GEV) -> np.ndarray:
    """Return lepton energy E (GeV) for given pt/eta and mass (massive-particle E)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px * px + py * py + pz * pz
    return np.sqrt(p2 + mass * mass)


def _jet_energy(pt: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Massless-ish jet energy E = |p| = pt*cosh(eta) (GeV)."""
    return pt * np.cosh(eta)


def _make_jets(n_events: int, rng: np.random.Generator, max_jets: int = 2):
    """Return (jet_n, jet_pt, jet_eta, jet_phi, jet_E) as ragged awkward arrays.

    Each list has ``jet_n[i]`` entries; energies/momenta are in GeV here (converted to
    MeV by the caller).
    """
    jet_n = rng.integers(0, max_jets + 1, size=n_events).astype(np.int32)
    pts, etas, phis, energies = [], [], [], []
    for nj in jet_n:
        if nj == 0:
            pts.append(np.empty(0))
            etas.append(np.empty(0))
            phis.append(np.empty(0))
            energies.append(np.empty(0))
            continue
        jpt = np.abs(rng.normal(40.0, 15.0, size=nj)) + 20.0  # >= 20 GeV jets
        jeta = rng.normal(0.0, 1.5, size=nj)
        jphi = rng.uniform(-np.pi, np.pi, size=nj)
        pts.append(jpt)
        etas.append(jeta)
        phis.append(jphi)
        energies.append(_jet_energy(jpt, jeta))
    return jet_n, pts, etas, phis, energies


def _ragged(list_of_arrays, scale: float = 1.0):
    """Build an awkward ragged array (one variable-length sublist per event)."""
    return ak.Array([np.asarray(a) * scale for a in list_of_arrays])


def _scalar_lepton_vector(values: np.ndarray, scale: float = 1.0):
    """A per-lepton branch where every event has exactly ONE lepton -> [[v0], [v1], ...]."""
    return ak.Array([[float(v) * scale] for v in values])


# --------------------------------------------------------------------------------------
# Event generators
# --------------------------------------------------------------------------------------
def make_signal(n: int, rng: np.random.Generator) -> dict:
    """W -> l nu -like events.

    * one lepton, pT ~ N(45, 10) GeV (clipped positive)
    * MET ~ N(35, 10) GeV
    * MET points roughly opposite the lepton in phi (back-to-back), which -- with these
      pT / MET scales -- drives mT up to its Jacobian edge near mW (~80 GeV).
    * lepton is well isolated (small ptcone30 / etcone20).
    """
    lep_pt = np.clip(rng.normal(45.0, 10.0, size=n), 5.0, None)
    lep_eta = rng.normal(0.0, 1.2, size=n)
    lep_phi = rng.uniform(-np.pi, np.pi, size=n)

    met_et = np.clip(rng.normal(35.0, 10.0, size=n), 0.5, None)
    # back-to-back with a bit of smearing so the mT edge is realistic, not a delta.
    met_phi = lep_phi + np.pi + rng.normal(0.0, 0.35, size=n)
    met_phi = np.arctan2(np.sin(met_phi), np.cos(met_phi))  # wrap to (-pi, pi]

    lep_type = rng.choice([TYPE_ELECTRON, TYPE_MUON], size=n).astype(np.int32)
    lep_charge = rng.choice([-1.0, 1.0], size=n)
    lep_isTightID = np.ones(n, dtype=np.int32)  # signal leptons pass tight ID

    # good isolation: cone activity is a small fraction of the lepton pT
    lep_ptcone30 = np.abs(rng.normal(0.0, 0.03, size=n)) * lep_pt
    lep_etcone20 = np.abs(rng.normal(0.0, 0.03, size=n)) * lep_pt

    lep_E = _lepton_four_vector(lep_pt, lep_eta, lep_phi)
    return _assemble(
        n, rng, lep_pt, lep_eta, lep_phi, lep_E, lep_type, lep_charge,
        lep_isTightID, lep_ptcone30, lep_etcone20, met_et, met_phi,
        xsection=10.0,  # pb; O(1) event weights with lumi/SumWeights below
    )


def make_background(n: int, rng: np.random.Generator) -> dict:
    """QCD / Z -like background.

    * one lepton, similar pT
    * fake/mismeasured MET with a tail (~N(24, 12) GeV) so a realistic fraction leaks
      past the met > 30 cut -- but the MET phi is uncorrelated with the lepton, so the
      transverse mass mT stays low and featureless (the key handle that separates it
      from genuine W -> l nu, whose lepton and MET are back-to-back).
    * a sizeable fraction of leptons are poorly isolated and/or fail tight ID.
    """
    lep_pt = np.clip(rng.normal(40.0, 12.0, size=n), 5.0, None)
    lep_eta = rng.normal(0.0, 1.4, size=n)
    lep_phi = rng.uniform(-np.pi, np.pi, size=n)

    met_et = np.clip(rng.normal(24.0, 12.0, size=n), 0.2, None)
    met_phi = rng.uniform(-np.pi, np.pi, size=n)  # uncorrelated fake MET

    lep_type = rng.choice([TYPE_ELECTRON, TYPE_MUON], size=n).astype(np.int32)
    lep_charge = rng.choice([-1.0, 1.0], size=n)
    # ~40% fail tight ID (fakes / non-prompt leptons)
    lep_isTightID = (rng.random(n) > 0.4).astype(np.int32)

    # worse isolation: a hadronic cone around the lepton. Mixture of ok and bad.
    frac = np.where(rng.random(n) > 0.5,
                    np.abs(rng.normal(0.25, 0.12, size=n)),   # poorly isolated
                    np.abs(rng.normal(0.05, 0.04, size=n)))   # some are ok
    lep_ptcone30 = frac * lep_pt
    lep_etcone20 = (frac * 0.9) * lep_pt

    lep_E = _lepton_four_vector(lep_pt, lep_eta, lep_phi)
    return _assemble(
        n, rng, lep_pt, lep_eta, lep_phi, lep_E, lep_type, lep_charge,
        lep_isTightID, lep_ptcone30, lep_etcone20, met_et, met_phi,
        xsection=50.0,  # larger cross-section -> more background, still O(1) weights
    )


def _assemble(n, rng, lep_pt, lep_eta, lep_phi, lep_E, lep_type, lep_charge,
              lep_isTightID, lep_ptcone30, lep_etcone20, met_et, met_phi,
              xsection: float) -> dict:
    """Pack one sample's branches into a dict of arrays, converting GeV -> MeV.

    Scalar energy branches (met_et) and per-lepton/per-jet energy branches
    (lep_pt, lep_E, jet_pt, jet_E) are stored in MeV (x1000).  Angles, etas, counts,
    charges, IDs and weights are unit-less and stored as-is.
    """
    jet_n, jpt, jeta, jphi, jE = _make_jets(n, rng)

    # SumWeights is the sum of generator weights over the *full* MC sample. With
    # mcWeight == 1 this is just the generated event count; choosing it here fixes the
    # event-weight normalisation (XSection * lumi / SumWeights) to be O(1).
    sum_weights = float(n)

    data = {
        # ---- scalars ----
        "lep_n": np.ones(n, dtype=np.int32),
        "met_et": met_et * MEV_PER_GEV,          # MeV
        "met_phi": met_phi.astype(np.float64),
        "jet_n": jet_n.astype(np.int32),
        # ---- MC truth / weight machinery ----
        "mcWeight": np.ones(n, dtype=np.float64),
        "XSection": np.full(n, xsection, dtype=np.float64),        # pb
        "SumWeights": np.full(n, sum_weights, dtype=np.float64),
        "scaleFactor_PILEUP": np.ones(n, dtype=np.float64),
        "scaleFactor_ELE": np.ones(n, dtype=np.float64),
        "scaleFactor_MUON": np.ones(n, dtype=np.float64),
        "scaleFactor_LepTRIGGER": np.ones(n, dtype=np.float64),
        # ---- per-lepton vectors (length lep_n == 1) ----
        "lep_pt": _scalar_lepton_vector(lep_pt, MEV_PER_GEV),      # MeV
        "lep_eta": _scalar_lepton_vector(lep_eta),
        "lep_phi": _scalar_lepton_vector(lep_phi),
        "lep_E": _scalar_lepton_vector(lep_E, MEV_PER_GEV),        # MeV
        "lep_type": _scalar_lepton_vector(lep_type),
        "lep_charge": _scalar_lepton_vector(lep_charge),
        "lep_isTightID": _scalar_lepton_vector(lep_isTightID),
        "lep_ptcone30": _scalar_lepton_vector(lep_ptcone30, MEV_PER_GEV),  # MeV
        "lep_etcone20": _scalar_lepton_vector(lep_etcone20, MEV_PER_GEV),  # MeV
        # ---- per-jet vectors (length jet_n) ----
        "jet_pt": _ragged(jpt, MEV_PER_GEV),      # MeV
        "jet_eta": _ragged(jeta),
        "jet_phi": _ragged(jphi),
        "jet_E": _ragged(jE, MEV_PER_GEV),        # MeV
    }
    return data


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------
def _write(path: str, data: dict) -> None:
    """Write a single TTree named 'mini' to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with uproot.recreate(path) as f:
        f["mini"] = data


def main() -> None:
    rng = np.random.default_rng(SEED)

    signal = make_signal(N_SIGNAL, rng)
    background = make_background(N_BKG, rng)

    sig_path = os.path.join(OUT_DIR, "sample_signal.root")
    bkg_path = os.path.join(OUT_DIR, "sample_bkg.root")

    _write(sig_path, signal)
    _write(bkg_path, background)

    print("Wrote synthetic ATLAS-Open-Data-style fixtures (TTree 'mini', MeV units):")
    print(f"  signal    -> {sig_path}  ({N_SIGNAL} events)")
    print(f"  background -> {bkg_path}  ({N_BKG} events)")
    print("Run the pipeline with:  python -m src.pipeline")


if __name__ == "__main__":
    main()
