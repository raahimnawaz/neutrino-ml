# neutrino-ml

Machine learning for **neutrinos** on real ATLAS Open Data (13 TeV LHC proton–proton
collisions), read from CERN ROOT `TTree`s with `RDataFrame`.

## The premise: a neutrino is missing energy

A neutrino leaves **no track** in ATLAS — it escapes the detector. Its only signature is an
imbalance in the transverse momentum of the event: **missing transverse energy** (`met_et`,
`met_phi`). So "do ML on neutrinos" means the two things collider physicists actually do:

- **Task A — detect it:** classify events that contain a genuine neutrino (a real `W→ℓν`
  decay) vs backgrounds where the MET is fake or mismeasured.
- **Task B — reconstruct it:** the neutrino's *transverse* momentum **is** the MET; the one
  unmeasured piece is its *longitudinal* momentum `pz`, recovered from the W-mass constraint
  (a quadratic → [`src/physics.py`](src/physics.py) `neutrino_pz`). This is the truest
  "predict the neutrino."

This repo implements Task A end-to-end and provides the Task-B reconstruction physics.

## Data

[ATLAS Open Data 13 TeV](https://opendata.atlas.cern/docs/data/for_education/13TeV_details) —
real 2016 pp collisions + matched MC, ROOT `TTree` named `mini`, ~10 fb⁻¹. Download the
`W→ℓν` signal samples and background samples (Z, ttbar, diboson) into `data/root/`
(gitignored). Key branches (all momenta/energies in **MeV**, converted to GeV on read):

| group | branches |
|---|---|
| lepton | `lep_n, lep_pt, lep_eta, lep_phi, lep_E, lep_type, lep_charge, lep_isTightID, lep_ptcone30, lep_etcone20` |
| neutrino (MET) | `met_et, met_phi` |
| jets | `jet_n, jet_pt, jet_eta, jet_phi, jet_E` |
| weights (MC) | `mcWeight, XSection, SumWeights, scaleFactor_*` |

## Setup

ROOT's Python bindings ship only for Linux/macOS, so on Windows run inside **WSL2** (see the
sibling `anubis-ml` project for the same pattern):

```bash
conda env create -f environment.yml      # ROOT + uproot + xgboost + sklearn + ...
conda activate neutrino-ml
```

## Quickstart — runs with NO real data

The pipeline is proven end-to-end against a **synthetic** ntuple, so you can run it before
downloading anything:

```bash
python tests/make_sample.py     # writes data/root/sample_{signal,bkg}.root (uproot, MeV units)
python -m src.pipeline          # read -> select -> weight -> train (XGBoost + logreg) -> evaluate
pytest tests/                   # physics unit tests (pure numpy, no ROOT)
```

Then drop real ATLAS files into `data/root/` and point `src/pipeline.py` at them.

## Pipeline

```
ROOT ntuple                       src/read_root.py    RDataFrame -> GeV arrays
   │
   ▼  W→ℓν selection              src/selection.py    1 tight isolated lepton, met>30, cutflow
   ▼  physical event weights      src/weights.py      mcWeight·SF·(XSection·lumi/SumWeights)
   ▼  engineered features         src/features.py     mt, dphi(ℓ,MET), HT, isolation, ...
   ▼  labelled dataset            src/dataset.py      -> data/processed/dataset.parquet
   ▼  models                      src/models.py       XGBoost + logistic-regression baseline
   ▼  physics evaluation          src/evaluation.py   weighted ROC/AUC, S/√B + Asimov Z, SIC
```

Everything is **weighted** — the split, the AUC, the significance — using the real MC event
weights. That weighting is the single most important thing separating a HEP analysis from a
generic classifier.

## Repo layout

```
src/
  physics.py       pure-numpy kinematics: transverse_mass, delta_phi, neutrino_pz (W-mass reco)
  read_root.py     RDataFrame reader (MeV -> GeV)
  selection.py     W→ℓν cuts + cutflow
  weights.py       physical MC event weights
  features.py      per-event feature table
  dataset.py       read+select+weight+featurize -> parquet
  models.py        XGBoost + logistic-regression
  evaluation.py    weighted AUC, significance scan, SIC, plots
  pipeline.py      end-to-end runner (python -m src.pipeline)
tests/
  test_physics.py  unit tests for the kinematics (run in CI, no ROOT)
  make_sample.py   synthetic ntuple generator for the smoke test
config.yaml        selection cuts, luminosity, feature list
```

## What's real vs simplified

**Real ATLAS methods here:** `RDataFrame` selection, physically-motivated features, the
transverse-mass handle, weighted significance (`S/√B` and the Asimov `Z`), ML-vs-baseline
comparison, and the neutrino-`pz` W-mass reconstruction.

**Simplified vs a publication:** no systematic uncertainties, no data-driven background
estimation (control regions), no profile-likelihood fit, and no blinding. Those are the bulk
of a real analysis — this is a faithful *miniature* of the workflow, not a stand-in for it.
