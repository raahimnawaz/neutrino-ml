"""End-to-end W -> l nu analysis pipeline (Task A: signal vs. background).

Wires the pieces together::

    build_dataset  ->  weighted train/test split  ->  train XGBoost + LogReg
                   ->  weighted evaluation (AUC, significance scan, SIC)
                   ->  save plots + best model  ->  print a readable summary

Run on the synthetic fixtures with::

    python tests/make_sample.py     # once, to create data/root/sample_*.root
    python -m src.pipeline          # reads config.yaml, globs the fixtures, runs

Everything is weighted: the train/test split, the AUC, the significance and the SIC
all carry the per-event MC weights, mirroring how a real ATLAS analysis quotes results
in terms of expected weighted event yields rather than raw counts.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src import evaluation
from src.dataset import build_dataset
from src.models import train_logreg, train_xgb

# Repo root (this file lives in <root>/src/), used to resolve config-relative paths.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
MODELS_DIR = os.path.join(REPO_ROOT, "saved_models")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _abspath(path: str) -> str:
    """Resolve a possibly-relative config path against the repo root."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _feature_matrix(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Select exactly the configured feature columns (order preserved)."""
    return df[list(cfg["features"])].copy()


def _pick(d: dict, *keys, default=float("nan")):
    """First present key from *keys* in dict *d* (tolerates naming variants)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _summarise_significance(name: str, scan: dict) -> str:
    """One-line human-readable summary of a significance_scan() result dict.

    The exact key names produced by ``evaluation.significance_scan`` are not pinned by
    the contract (only that it reports the best threshold, the best ``Z=S/sqrt(B)`` and
    the Asimov ``Z``), so we look up a few sensible aliases and never crash on a miss.
    """
    best_z = _pick(scan, "best_z", "best_significance", "z", "significance")
    thr = _pick(scan, "best_threshold", "threshold", "cut")
    asimov = _pick(scan, "asimov", "z_asimov", "best_asimov", "asimov_z")
    s = _pick(scan, "s_at_best", "s", "S", "signal", "signal_yield")
    b = _pick(scan, "b_at_best", "b", "B", "background", "background_yield")
    return (
        f"  {name:8s}  best Z(S/sqrt(B))={best_z:.3f} @ threshold={thr:.3f} | "
        f"Z_asimov={asimov:.3f} (S={s:.1f}, B={b:.1f})"
    )


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------
def run(signal_files, bkg_files, cfg: dict) -> dict:
    """Run the full Task-A pipeline and return a results dictionary.

    Parameters
    ----------
    signal_files, bkg_files:
        Lists of ROOT file paths for the W -> l nu signal and the background.
    cfg:
        The parsed ``config.yaml`` dictionary.

    Returns
    -------
    dict
        ``{"dataset", "models", "auc", "significance", "sic", "plots", "model_path"}``.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- 1. build the (already selected + weighted) dataset --------------------------
    df = build_dataset(signal_files, bkg_files, cfg)
    n_sig = int((df["label"] == 1).sum())
    n_bkg = int((df["label"] == 0).sum())
    print(f"[pipeline] dataset: {len(df)} selected events "
          f"({n_sig} signal, {n_bkg} background)")
    if len(df) == 0:
        raise RuntimeError("Empty dataset after selection -- check inputs / cuts.")

    X = _feature_matrix(df, cfg)
    y = df["label"].to_numpy().astype(int)
    w = df["weight"].to_numpy().astype(float)

    # --- 2. weighted, stratified train/test split ------------------------------------
    test_size = float(cfg["train"]["test_size"])
    random_state = int(cfg["train"]["random_state"])
    (X_tr, X_te, y_tr, y_te, w_tr, w_te) = train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[pipeline] split: {len(X_tr)} train / {len(X_te)} test "
          f"(test_size={test_size}, stratified by label)")

    # --- 3. train the classifiers ----------------------------------------------------
    print("[pipeline] training XGBoost ...")
    xgb_model = train_xgb(X_tr, y_tr, w_tr)
    print("[pipeline] training logistic-regression baseline ...")
    logreg_model = train_logreg(X_tr, y_tr, w_tr)

    models = {"xgb": xgb_model, "logreg": logreg_model}

    # --- 4. evaluate on the held-out test set (all weighted) -------------------------
    def _scores(model) -> np.ndarray:
        # signal-class probability
        return model.predict_proba(X_te)[:, 1]

    results: dict[str, Any] = {
        "auc": {}, "significance": {}, "sic": {}, "plots": {}, "scores": {},
    }
    for name, model in models.items():
        scores = _scores(model)
        results["scores"][name] = scores
        results["auc"][name] = evaluation.weighted_auc(y_te, scores, w_te)
        results["significance"][name] = evaluation.significance_scan(y_te, scores, w_te)
        eff, sic = evaluation.sic_curve(y_te, scores, w_te)
        results["sic"][name] = {"signal_eff": eff, "sic": sic, "max_sic": float(np.nanmax(sic)) if len(sic) else float("nan")}

    # --- 5. pick the better model by weighted AUC, save plots + model ----------------
    best_name = max(results["auc"], key=results["auc"].get)
    best_model = models[best_name]
    best_scores = results["scores"][best_name]

    roc_path = os.path.join(RESULTS_DIR, "roc.png")
    sig_path = os.path.join(RESULTS_DIR, "significance.png")
    evaluation.plot_roc(y_te, best_scores, w_te, roc_path)
    evaluation.plot_significance(y_te, best_scores, w_te, sig_path)
    results["plots"] = {"roc": roc_path, "significance": sig_path}

    model_path = os.path.join(MODELS_DIR, "classifier.joblib")
    joblib.dump({"model": best_model, "features": list(cfg["features"]),
                 "name": best_name}, model_path)

    results["dataset"] = df
    results["models"] = models
    results["best_model"] = best_name
    results["model_path"] = model_path

    # --- 6. readable summary ---------------------------------------------------------
    print("\n" + "=" * 66)
    print("  neutrino-ml  |  W -> l nu  classification summary (Task A)")
    print("=" * 66)
    print(f"  events (weighted): S={w[y == 1].sum():.1f}  B={w[y == 0].sum():.1f}")
    print("-" * 66)
    print("  weighted ROC AUC:")
    for name in models:
        star = "  <-- selected" if name == best_name else ""
        print(f"    {name:8s} {results['auc'][name]:.4f}{star}")
    print("-" * 66)
    print("  significance (S / sqrt(B)) scan + Asimov Z:")
    for name in models:
        print(_summarise_significance(name, results["significance"][name]))
    print("-" * 66)
    print("  max SIC (tpr / sqrt(fpr)):")
    for name in models:
        print(f"    {name:8s} {results['sic'][name]['max_sic']:.3f}")
    print("-" * 66)
    print(f"  saved model -> {model_path}")
    print(f"  saved plots -> {roc_path}")
    print(f"                 {sig_path}")
    print("=" * 66 + "\n")

    return results


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg_path = os.path.join(REPO_ROOT, "config.yaml")
    cfg = _load_config(cfg_path)

    root_dir = _abspath(cfg["data"]["root_dir"])
    signal_files = sorted(glob.glob(os.path.join(root_dir, "sample_signal*.root")))
    bkg_files = sorted(glob.glob(os.path.join(root_dir, "sample_bkg*.root")))

    if not signal_files or not bkg_files:
        raise FileNotFoundError(
            f"No sample fixtures found in {root_dir}. "
            "Generate them first with:  python tests/make_sample.py"
        )

    print(f"[pipeline] signal files: {signal_files}")
    print(f"[pipeline] background files: {bkg_files}")
    run(signal_files, bkg_files, cfg)


if __name__ == "__main__":
    main()
