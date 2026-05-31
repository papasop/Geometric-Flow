#!/usr/bin/env python3
"""
Generate openrouter-usage.json for the OpenRouter weekly token usage chart.

The public page reads openrouter-usage.json first and falls back to the inline
HTML seed. The seed is transcribed from the Semafor chart credited to
Marta Biino, with OpenRouter as the source. A real weekly Anthropic + ChatGPT
/ OpenAI value can be supplied with --tokens or OPENROUTER_WEEKLY_TOKENS.
Otherwise the updater searches public OpenRouter/Semafor mentions with Tavily;
if no parseable filtered value is found, it carries the latest known value
forward and marks that in metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTFILE = ROOT / "openrouter-usage.json"
MIRROR_DIRS = [ROOT / "agi", ROOT / "ai-global-index"]
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
USER_AGENT = "IsItHub/1.0 (openrouter-usage; https://isithub.com/)"
SOURCE_NAME = "Tavily public search: OpenRouter Anthropic + ChatGPT/OpenAI weekly token usage"
SOURCE_URL = "https://openrouter.ai/rankings"
MODEL_SCOPE = "anthropic_chatgpt_openai"
MODEL_FAMILIES = {
    "anthropic": {
        "label": "Anthropic / Claude",
        "terms": ("anthropic", "claude", "opus", "sonnet", "haiku"),
        "queries": [
            '"OpenRouter" "Anthropic" "token usage" "trillion"',
            '"OpenRouter" "Claude" "tokens" "Semafor"',
            '"OpenRouter" "Claude" "weekly tokens"',
            '"OpenRouter rankings" "Anthropic" "tokens"',
        ],
    },
    "openai": {
        "label": "ChatGPT / OpenAI",
        "terms": ("openai", "chatgpt", "gpt-4", "gpt-5", "gpt 4", "gpt 5", "o3", "o4"),
        "queries": [
            '"OpenRouter" "ChatGPT" "token usage" "trillion"',
            '"OpenRouter" "OpenAI" "tokens" "Semafor"',
            '"OpenRouter" "GPT-5" "weekly tokens"',
            '"OpenRouter rankings" "OpenAI" "tokens"',
        ],
    },
}

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


def parse_token_value(text: str) -> float | None:
    cleaned = re.sub(r"\s+", " ", text or "")
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:T|tn|trillion)\s+(?:tokens|token)",
        r"(\d+(?:\.\d+)?)\s*(?:T|tn|trillion)\b",
        r"(\d+(?:\.\d+)?)\s*(?:万亿|兆)\s*(?:tokens|token|代币|令牌)?",
    ]
    values: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.I):
            value = float(match.group(1)) * 1e12
            if 1e12 <= value <= 200e12:
                values.append(value)
    if not values:
        return None
    return round(max(values), 2)


def parse_family_token_value(text: str, family_terms: tuple[str, ...]) -> float | None:
    """Parse token usage values only when they appear near a target model family."""
    cleaned = re.sub(r"\s+", " ", text or "")
    lowered = cleaned.lower()
    if not any(term in lowered for term in family_terms):
        return None
    term_positions: list[int] = []
    for term in family_terms:
        term_positions.extend(match.start() for match in re.finditer(re.escape(term), lowered))

    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:T|tn|trillion)\s+(?:tokens|token)",
        r"(\d+(?:\.\d+)?)\s*(?:T|tn|trillion)\b",
        r"(\d+(?:\.\d+)?)\s*(?:万亿|兆)\s*(?:tokens|token|代币|令牌)?",
    ]
    candidates: list[tuple[int, float]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.I):
            start = max(0, match.start() - 180)
            end = min(len(cleaned), match.end() + 180)
            window = cleaned[start:end].lower()
            if not any(term in window for term in family_terms):
                continue
            before_start = max(0, match.start() - 52)
            before_window = cleaned[before_start:match.start()].lower()
            if "all models" in before_window or "across models" in before_window:
                continue
            value = float(match.group(1)) * 1e12
            if 1e9 <= value <= 200e12:
                distance = min(abs(match.start() - pos) for pos in term_positions)
                candidates.append((distance, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return round(candidates[0][1], 2)


def post_tavily_search(api_key: str, query: str) -> dict:
    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": 5,
        "time_range": "month",
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
    }
    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        TAVILY_SEARCH_ENDPOINT,
        data=encoded,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def search_family_weekly_tokens(api_key: str, family_key: str, config: dict[str, object]) -> dict[str, object]:
    terms = tuple(str(term).lower() for term in config["terms"])
    observations: list[dict[str, object]] = []
    for query in config["queries"]:
        try:
            data = post_tavily_search(api_key, str(query))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            observations.append({"query": query, "status": "error", "error": str(exc)[:180]})
            continue

        texts: list[str] = []
        answer = data.get("answer") if isinstance(data, dict) else ""
        if answer:
            texts.append(str(answer))
        results = data.get("results") if isinstance(data, dict) else []
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                texts.append(" ".join([
                    str(result.get("title") or ""),
                    str(result.get("content") or ""),
                    str(result.get("url") or ""),
                ]))

        for text in texts:
            value = parse_family_token_value(text, terms)
            if value:
                observations.append({"query": query, "status": "parsed", "tokens": value})
                return {"family": family_key, "tokens": value, "status": "parsed", "observations": observations}
        observations.append({"query": query, "status": "no-parse"})

    return {"family": family_key, "tokens": None, "status": "no-public-family-value", "observations": observations}


def search_public_weekly_tokens() -> dict[str, object]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"tokens": None, "status": "missing-key", "observations": []}

    family_results: dict[str, dict[str, object]] = {}
    family_tokens: dict[str, float] = {}
    for family_key, config in MODEL_FAMILIES.items():
        result = search_family_weekly_tokens(api_key, family_key, config)
        family_results[family_key] = result
        if result.get("tokens"):
            family_tokens[family_key] = round(float(result["tokens"]), 2)

    if set(family_tokens) == set(MODEL_FAMILIES):
        total = round(sum(family_tokens.values()), 2)
        return {
            "tokens": total,
            "status": "parsed",
            "scope": MODEL_SCOPE,
            "familyTokens": family_tokens,
            "observations": family_results,
        }

    return {
        "tokens": None,
        "status": "missing-family-values",
        "scope": MODEL_SCOPE,
        "familyTokens": family_tokens,
        "observations": family_results,
    }


def resolve_tokens(cli_tokens: float | None, previous_value: float) -> tuple[float, str, dict[str, object] | None]:
    raw = cli_tokens if cli_tokens is not None else os.environ.get("OPENROUTER_WEEKLY_TOKENS")
    if raw not in (None, ""):
        value = float(raw)
        if value <= 0:
            raise ValueError("OpenRouter weekly tokens must be positive")
        return round(value, 2), "manual", None

    search = search_public_weekly_tokens()
    if search.get("tokens"):
        return round(float(search["tokens"]), 2), "tavily-filtered-public-search", search
    return round(previous_value, 2), "carry-forward-filtered-scope", search


def update_usage(target_day: date, tokens: float | None) -> dict:
    rows = read_existing_rows()
    target_week = week_start(target_day)
    last_week = parse_day(rows[-1][0])
    last_value = float(rows[-1][1])
    update_mode = "unchanged"
    search_status: dict[str, object] | None = None

    if target_week < last_week:
        raise ValueError(f"target week {target_week} is before latest row {last_week}")
    if target_week == last_week:
        rows[-1][1], update_mode, search_status = resolve_tokens(tokens, last_value)
    else:
        current = last_week + timedelta(days=7)
        while current <= target_week:
            if current == target_week:
                next_value, update_mode, search_status = resolve_tokens(tokens, last_value)
            else:
                next_value = last_value
            rows.append([current.isoformat(), next_value])
            last_value = next_value
            current += timedelta(days=7)

    latest = rows[-1]
    return {
        "meta": {
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "source": SOURCE_NAME,
            "sourceUrl": SOURCE_URL,
            "scope": MODEL_SCOPE,
            "families": {key: config["label"] for key, config in MODEL_FAMILIES.items()},
            "unit": "tokens per week",
            "count": len(rows),
            "latestWeek": latest[0],
            "latestValue": latest[1],
            "updateMode": update_mode,
            "carriedForward": update_mode.startswith("carry-forward"),
            "searchStatus": search_status,
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
