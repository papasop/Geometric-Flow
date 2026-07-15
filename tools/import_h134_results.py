#!/usr/bin/env python3
"""
Import H13.4 Colab outputs into a repository without inventing results.

Example:
    python tools/import_h134_results.py \
        --source ~/Downloads/geoflow_h134_full_product_audit_results \
        --repo .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED = [
    "h134_stepwise_full_product_metrics.csv",
    "h134_stepwise_full_product_metrics.json",
    "h134_full_product_summary.csv",
    "h134_full_product_summary.json",
    "h134_full_product_optimizer_contrast.csv",
    "h134_full_product_optimizer_contrast.json",
    "h134_full_product_branch_conditions.csv",
    "h134_config.json",
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
    parser.add_argument(
        "--destination",
        default="results/h134",
        help="Destination relative to repository root.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Copy available files even if some expected files are absent.",
    )
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
            "Missing expected H13.4 outputs:\n  - " + "\n  - ".join(missing)
        )

    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "H13.4",
        "source": str(source),
        "destination": str(destination),
        "copied": [],
        "missing": missing,
    }

    for name in available:
        src = source / name
        dst = destination / name
        shutil.copy2(src, dst)
        manifest["copied"].append(
            {
                "filename": name,
                "sha256": sha256(dst),
                "bytes": dst.stat().st_size,
            }
        )
        print(f"copied {src} -> {dst}")

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
