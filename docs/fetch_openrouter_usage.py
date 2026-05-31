#!/usr/bin/env python3
"""
Generate openrouter-usage.json for the OpenRouter weekly token usage chart.

The public page reads openrouter-usage.json first and falls back to the inline
HTML seed. The seed is transcribed from the Semafor chart credited to
Marta Biino, with OpenRouter as the source. A real weekly value can be supplied
with --tokens or OPENROUTER_WEEKLY_TOKENS; otherwise the updater carries the
latest known value forward so the feed metadata refreshes daily.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTFILE = ROOT / "openrouter-usage.json"
MIRROR_DIRS = [ROOT / "agi", ROOT / "ai-global-index"]

SEED_ROWS: list[list[object]] = [
    ["2025-10-13", 5.2e12],
    ["2025-10-20", 4.9e12],
    ["2025-10-27", 4.8e12],
    ["2025-11-03", 5.4e12],
    ["2025-11-10", 5.8e12],
    ["2025-11-17", 5.8e12],
    ["2025-11-24", 6.1e12],
    ["2025-12-01", 7.4e12],
    ["2025-12-08", 6.3e12],
    ["2025-12-15", 5.8e12],
    ["2025-12-22", 5.8e12],
    ["2025-12-29", 5.7e12],
    ["2026-01-05", 5.5e12],
    ["2026-01-12", 6.4e12],
    ["2026-01-19", 7.7e12],
    ["2026-01-26", 7.4e12],
    ["2026-02-02", 8.1e12],
    ["2026-02-09", 9.8e12],
    ["2026-02-16", 13.0e12],
    ["2026-02-23", 14.0e12],
    ["2026-03-02", 13.6e12],
    ["2026-03-09", 14.9e12],
    ["2026-03-16", 16.8e12],
    ["2026-03-23", 20.4e12],
    ["2026-03-30", 22.7e12],
    ["2026-04-06", 26.9e12],
    ["2026-04-13", 21.0e12],
    ["2026-04-20", 20.5e12],
]


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def normalize_rows(rows) -> list[list[object]]:
    by_date: dict[str, float] = {}
    for row in rows or []:
        if isinstance(row, dict):
            raw_date = row.get("date") or row.get("week")
            raw_value = row.get("tokens") or row.get("value")
        else:
            raw_date = row[0] if len(row) > 0 else None
            raw_value = row[1] if len(row) > 1 else None
        try:
            day = parse_day(str(raw_date)).isoformat()
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            by_date[day] = round(value, 2)
    return [[day, by_date[day]] for day in sorted(by_date)]


def read_existing_rows() -> list[list[object]]:
    if not OUTFILE.exists():
        return normalize_rows(SEED_ROWS)
    payload = json.loads(OUTFILE.read_text(encoding="utf-8"))
    rows = payload.get("usage") if isinstance(payload, dict) else payload
    clean = normalize_rows(rows)
    return clean or normalize_rows(SEED_ROWS)


def resolve_tokens(cli_tokens: float | None, previous_value: float) -> float:
    raw = cli_tokens if cli_tokens is not None else os.environ.get("OPENROUTER_WEEKLY_TOKENS")
    if raw not in (None, ""):
        value = float(raw)
        if value <= 0:
            raise ValueError("OpenRouter weekly tokens must be positive")
        return round(value, 2)
    return round(previous_value, 2)


def update_usage(target_day: date, tokens: float | None) -> dict:
    rows = read_existing_rows()
    target_week = week_start(target_day)
    last_week = parse_day(rows[-1][0])
    last_value = float(rows[-1][1])

    if target_week < last_week:
        raise ValueError(f"target week {target_week} is before latest row {last_week}")
    if target_week == last_week:
        if tokens is not None or os.environ.get("OPENROUTER_WEEKLY_TOKENS") not in (None, ""):
            rows[-1][1] = resolve_tokens(tokens, last_value)
    else:
        current = last_week + timedelta(days=7)
        while current <= target_week:
            next_value = resolve_tokens(tokens, last_value) if current == target_week else last_value
            rows.append([current.isoformat(), next_value])
            last_value = next_value
            current += timedelta(days=7)

    return {
        "meta": {
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "source": "Semafor chart by Marta Biino / OpenRouter weekly token usage",
            "sourceUrl": "https://openrouter.ai/rankings",
            "unit": "tokens per week",
            "count": len(rows),
        },
        "usage": rows,
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
    parser.add_argument("--tokens", type=float, default=None, help="Optional weekly token usage for the target week.")
    args = parser.parse_args()

    payload = update_usage(parse_day(args.date), args.tokens)
    write_payload(payload)
    latest = payload["usage"][-1]
    print(f"openrouter-usage.json updated: week {latest[0]} = {latest[1]:.0f} tokens")


if __name__ == "__main__":
    main()
