#!/usr/bin/env python3
"""
Generate token-index.json for the LLM Token Expenditure Index chart.

The page reads token-index.json first and falls back to the inline HTML
constant when this feed is unavailable. By default, the updater appends missing
calendar days by carrying forward the latest known index value. A real daily
value can be supplied with --value or TOKEN_INDEX_VALUE.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
OUTFILE = ROOT / "token-index.json"
MIRROR_DIRS = [ROOT / "agi", ROOT / "ai-global-index"]


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def read_inline_index() -> list[list[object]]:
    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r"const\s+TOKEN_EXPENDITURE_INDEX\s*=\s*(\[[\s\S]*?\]);", html)
    if not match:
        raise RuntimeError("TOKEN_EXPENDITURE_INDEX not found in index.html")
    rows = json.loads(match.group(1))
    return normalize_rows(rows)


def normalize_rows(rows) -> list[list[object]]:
    by_date: dict[str, float] = {}
    for row in rows or []:
        if isinstance(row, dict):
            raw_date = row.get("date")
            raw_value = row.get("value")
        else:
            raw_date = row[0] if len(row) > 0 else None
            raw_value = row[1] if len(row) > 1 else None
        try:
            day = parse_day(str(raw_date)).isoformat()
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            by_date[day] = round(value, 4)
    return [[day, by_date[day]] for day in sorted(by_date)]


def read_existing_index() -> list[list[object]]:
    if not OUTFILE.exists():
        return read_inline_index()
    payload = json.loads(OUTFILE.read_text(encoding="utf-8"))
    rows = payload.get("index") if isinstance(payload, dict) else payload
    clean = normalize_rows(rows)
    return clean or read_inline_index()


def resolve_value(cli_value: float | None, previous_value: float) -> float:
    env_value = os.environ.get("TOKEN_INDEX_VALUE")
    raw = cli_value if cli_value is not None else env_value
    if raw not in (None, ""):
        value = float(raw)
        if value <= 0:
            raise ValueError("Token index value must be positive")
        return round(value, 4)
    return round(previous_value, 4)


def update_index(target_day: date, value: float | None) -> dict:
    rows = read_existing_index()
    if not rows:
        raise RuntimeError("No token index seed rows available")

    last_day = parse_day(rows[-1][0])
    last_value = float(rows[-1][1])
    if target_day < last_day:
        raise ValueError(f"target day {target_day} is before latest row {last_day}")
    if target_day == last_day:
        if value is not None or os.environ.get("TOKEN_INDEX_VALUE") not in (None, ""):
            rows[-1][1] = resolve_value(value, last_value)
        return {
            "meta": {
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "source": "Manual/carry-forward LLM Token Expenditure Index feed",
                "count": len(rows),
            },
            "index": rows,
        }

    current = last_day + timedelta(days=1)
    while current <= target_day:
        next_value = resolve_value(value, last_value) if current == target_day else last_value
        rows.append([current.isoformat(), next_value])
        last_value = next_value
        current += timedelta(days=1)

    return {
        "meta": {
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "source": "Manual/carry-forward LLM Token Expenditure Index feed",
            "count": len(rows),
        },
        "index": rows,
    }


def write_payload(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    OUTFILE.write_text(text, encoding="utf-8")
    for mirror in MIRROR_DIRS:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / OUTFILE.name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat(), help="Target YYYY-MM-DD date; defaults to today.")
    parser.add_argument("--value", type=float, default=None, help="Optional real index value for the target date.")
    args = parser.parse_args()

    payload = update_index(parse_day(args.date), args.value)
    write_payload(payload)
    latest = payload["index"][-1]
    print(f"token-index.json updated: {latest[0]} = {latest[1]:.4f}")


if __name__ == "__main__":
    main()
