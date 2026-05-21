#!/usr/bin/env python3
"""Refresh the local model ranking feed used by the static dashboard."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
OUTFILE = ROOT / "model-rankings.json"
MIRRORS = [
    ROOT / "agi" / "model-rankings.json",
    ROOT / "ai-global-index" / "model-rankings.json",
]


def normalize_rows(rows: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return normalized
    for row in rows:
        if isinstance(row, list) and len(row) >= 4:
            model, provider, score, cost = row[:4]
        elif isinstance(row, dict):
            model = row.get("model") or row.get("name")
            provider = row.get("provider", "")
            score = row.get("score")
            cost = row.get("avgCostPerTask", row.get("cost", row.get("avg_cost_per_task")))
        else:
            continue
        try:
            score_num = float(score)
            cost_num = float(cost)
        except (TypeError, ValueError):
            continue
        if not model or score_num < 0 or cost_num < 0:
            continue
        normalized.append(
            {
                "model": str(model),
                "provider": str(provider),
                "score": round(score_num, 2),
                "avgCostPerTask": round(cost_num, 4),
            }
        )
    return normalized


def read_inline_rankings() -> list[dict[str, Any]]:
    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r"const\s+MODEL_RANKINGS\s*=\s*(\[[\s\S]*?\]);", html)
    if not match:
        return []
    return normalize_rows(json.loads(match.group(1)))


def read_existing_rankings() -> list[dict[str, Any]]:
    if not OUTFILE.exists():
        return []
    try:
        payload = json.loads(OUTFILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload.get("rankings") if isinstance(payload, dict) else payload
    return normalize_rows(rows)


def write_feed(rankings: list[dict[str, Any]]) -> None:
    payload = {
        "meta": {
            "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source": "IsitHUB model ranking feed",
            "count": len(rankings),
        },
        "rankings": rankings,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    OUTFILE.write_text(text, encoding="utf-8")
    for mirror in MIRRORS:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(text, encoding="utf-8")


def main() -> None:
    rankings = read_inline_rankings() or read_existing_rankings()
    if not rankings:
        raise SystemExit("No model rankings available")
    write_feed(rankings)
    print(f"Wrote {len(rankings)} model rankings to {OUTFILE}")


if __name__ == "__main__":
    main()
