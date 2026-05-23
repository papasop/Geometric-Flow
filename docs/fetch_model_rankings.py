#!/usr/bin/env python3
"""Refresh the local model ranking feed used by the static dashboard."""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
OUTFILE = ROOT / "model-rankings.json"
CURSORBENCH_URL = "https://cursor.com/cursorbench"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
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


def infer_provider(model: str) -> str:
    text = model.lower()
    if "opus" in text or "claude" in text:
        return "anthropic"
    if "gpt" in text or text.startswith(("o3", "o4")):
        return "openai"
    if "composer" in text:
        return "cursor"
    if "gemini" in text:
        return "google"
    if "kimi" in text:
        return "moonshot"
    return ""


def parse_cursorbench_html(html: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r'aria-label="([^"]+?):\s*([0-9]+(?:\.[0-9]+)?)%,\s*'
        r'\$([0-9]+(?:\.[0-9]+)?)\s+avg cost per task"',
        re.IGNORECASE,
    )
    for match in pattern.finditer(html):
        model = unescape(match.group(1)).strip()
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "model": model,
                "provider": infer_provider(model),
                "score": float(match.group(2)),
                "avgCostPerTask": float(match.group(3)),
            }
        )
    rankings = normalize_rows(rows)
    rankings.sort(key=lambda row: (-row["score"], row["avgCostPerTask"], row["model"]))
    return rankings


def fetch_cursorbench_rankings() -> list[dict[str, Any]]:
    request = urllib.request.Request(CURSORBENCH_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        print(f"CursorBench fetch failed: {exc}")
        return []
    return parse_cursorbench_html(html)


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


def write_feed(rankings: list[dict[str, Any]], source: str, source_url: str = "") -> None:
    payload = {
        "meta": {
            "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source": source,
            "sourceUrl": source_url,
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
    rankings = fetch_cursorbench_rankings()
    source = "CursorBench 3.1"
    source_url = CURSORBENCH_URL
    if not rankings:
        rankings = read_existing_rankings() or read_inline_rankings()
        source = "IsitHUB model ranking fallback"
        source_url = ""
    if not rankings:
        raise SystemExit("No model rankings available")
    write_feed(rankings, source, source_url)
    print(f"Wrote {len(rankings)} model rankings to {OUTFILE} from {source}")


if __name__ == "__main__":
    main()
