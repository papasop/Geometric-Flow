#!/usr/bin/env python3
"""Import genuine H13.5 Colab outputs and generate a checksum manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED = [
    "h135_stepwise_counterfactual_metrics.csv",
    "h135_stepwise_counterfactual_metrics.json",
    "h135_counterfactual_summary.csv",
    "h135_counterfactual_summary.json",
    "h135_counterfactual_contrast.csv",
    "h135_counterfactual_contrast.json",
    "h135_counterfactual_branch_conditions.csv",
    "h135_config.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repo", default=Path("."), type=Path)
    parser.add_argument("--destination", default="results/h135")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    destination = (repo / args.destination).resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git repository root: {repo}")

    available = [name for name in EXPECTED if (source / name).is_file()]
    missing = [name for name in EXPECTED if name not in available]
    if missing and not args.allow_partial:
        raise SystemExit(
            "Missing expected H13.5 outputs:\n  - " + "\n  - ".join(missing)
        )

    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in available:
        src = source / name
        dst = destination / name
        shutil.copy2(src, dst)
        copied.append(
            {
                "filename": name,
                "bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
        )
        print(f"copied {src} -> {dst}")

    manifest = {
        "experiment": "H13.5",
        "source": str(source),
        "destination": str(destination),
        "copied": copied,
        "missing": missing,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
