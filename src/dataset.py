"""Assemble the labelled training dataset for the W -> l nu classifier (Task A).

For each input group (signal = W -> l nu, background = QCD/Z-like) this module runs the
full front-end of the analysis:

    read (ROOT ntuple -> GeV arrays)  ->  select (W -> l nu cuts)
        ->  weight (per-event MC weight)  ->  featurize (flat feature table)

then labels the rows (1 = signal, 0 = background), attaches the per-event weight,
concatenates the two groups, persists the result to Parquet, and returns it.

The output columns are ``features.FEATURE_COLUMNS`` plus ``label`` and ``weight``.
Weights are computed over ALL events in a group and then subset to the selected rows,
so the surviving-event weights correctly reflect the cross-section normalisation.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import features as features_mod
from . import read_root
from . import selection as selection_mod
from . import weights as weights_mod


def _process_group(
    files: Sequence[str],
    label: int,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Read one file group, apply selection, and return its labelled feature table.

    Weights are evaluated on the full (pre-selection) event set and then masked down
    to the selected events, matching the selection mask returned by ``selection.select``.
    """
    tree = cfg.get("data", {}).get("tree", "mini")
    lumi_pb = float(cfg["data"]["lumi_pb"])

    events = read_root.read_events(list(files), tree=tree)

    # Sequential W -> l nu selection. cutflow is (name, surviving_count) tuples.
    mask, cutflow = selection_mod.select(events, cfg)
    mask = np.asarray(mask, dtype=bool)

    # Per-event weights over the full group, then restrict to selected events.
    w_all = weights_mod.event_weights(events, lumi_pb)
    w_sel = np.asarray(w_all, dtype=float)[mask]

    # Flat feature table for the selected events (rows aligned with w_sel).
    df = features_mod.build_features(events, mask, cfg)
    if len(df) != len(w_sel):
        raise ValueError(
            f"feature rows ({len(df)}) do not match selected weights ({len(w_sel)})"
        )

    df["label"] = int(label)
    df["weight"] = w_sel

    group = "signal" if label == 1 else "background"
    final = cutflow[-1][1] if cutflow else len(df)
    print(f"[dataset] {group}: {final} events pass selection from {len(files)} file(s)")

    return df


def build_dataset(
    signal_files: Sequence[str],
    bkg_files: Sequence[str],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Build, persist, and return the full labelled dataset.

    Parameters
    ----------
    signal_files, bkg_files
        Lists of ROOT file paths for the signal (W -> l nu) and background samples.
    cfg
        Parsed ``config.yaml`` mapping. Uses ``data.lumi_pb``, ``data.processed_dir``,
        the ``selection`` block, and (optionally) ``data.tree``.

    Returns
    -------
    pandas.DataFrame
        Columns = ``features.FEATURE_COLUMNS`` + ``["label", "weight"]``. Signal rows
        first, then background, with a fresh 0..N-1 index. Also written to
        ``{data.processed_dir}/dataset.parquet``.
    """
    sig_df = _process_group(signal_files, label=1, cfg=cfg)
    bkg_df = _process_group(bkg_files, label=0, cfg=cfg)

    dataset = pd.concat([sig_df, bkg_df], ignore_index=True)

    # Guarantee a stable, contract-matching column order: features, then label, weight.
    ordered_cols = list(features_mod.FEATURE_COLUMNS) + ["label", "weight"]
    dataset = dataset[ordered_cols]

    processed_dir = cfg["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "dataset.parquet")
    dataset.to_parquet(out_path, index=False)

    n_sig = int((dataset["label"] == 1).sum())
    n_bkg = int((dataset["label"] == 0).sum())
    print(
        f"[dataset] wrote {len(dataset)} events "
        f"({n_sig} signal / {n_bkg} background) -> {out_path}"
    )

    return dataset
