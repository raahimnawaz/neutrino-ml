"""Download the ATLAS Open Data 13 TeV reduced W dataset (1-lepton skim) locally.

Files are streamed from CERN's public EOS over XRootD and copied into ``data/root/``:

    data/root/mc/<name>.1lep.root     (all MC processes)
    data/root/data/<name>.1lep.root   (the 4 real-data periods)

The whole reduced dataset is only ~60 MB, so by default we grab everything. Already-
present files are skipped, so re-running just fills in what's missing.

Usage:
    python scripts/download_data.py                 # everything (~60 MB)
    python scripts/download_data.py wjets zjets      # only these process categories
    python scripts/download_data.py --list           # print the file plan, download nothing

Requires the XRootD-enabled ROOT from environment.yml (already used by the pipeline).
"""

from __future__ import annotations

import json
import os
import sys

import ROOT

BASE = "root://eospublic.cern.ch//eos/root-eos/reduced_atlas_opendata/w/1lep"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SAMPLES = json.load(open(os.path.join(HERE, "samples.json")))


def plan(categories: list[str]) -> list[tuple[str, str, str]]:
    """Return (category, remote_url, local_path) for each file to fetch."""
    items = []
    for cat, names in SAMPLES.items():
        if cat.startswith("_") or (categories and cat not in categories):
            continue
        remote_sub = "Data" if cat == "data" else "MC"
        local_sub = "data" if cat == "data" else "mc"
        for name in names:
            url = f"{BASE}/{remote_sub}/{name}.1lep.root"
            dst = os.path.join(REPO, "data", "root", local_sub, f"{name}.1lep.root")
            items.append((cat, url, dst))
    return items


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    list_only = "--list" in sys.argv
    items = plan(args)

    if list_only:
        for cat, url, dst in items:
            print(f"[{cat:9s}] {url}")
        print(f"\n{len(items)} files.")
        return

    os.makedirs(os.path.join(REPO, "data", "root", "mc"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "data", "root", "data"), exist_ok=True)

    ok, skipped, failed = 0, 0, 0
    for i, (cat, url, dst) in enumerate(items, 1):
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skipped += 1
            continue
        # ROOT.TFile.Cp handles the XRootD -> local copy.
        if ROOT.TFile.Cp(url, dst, False):
            ok += 1
            print(f"[{i}/{len(items)}] {cat}: {os.path.basename(dst)}")
        else:
            failed += 1
            print(f"[{i}/{len(items)}] FAILED {url}")

    print(f"\ndone: {ok} downloaded, {skipped} already present, {failed} failed")


if __name__ == "__main__":
    main()
