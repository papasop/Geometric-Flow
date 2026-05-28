#!/usr/bin/env python3
"""Build a Tavily-backed talent-flow graph for IsItHub.

The public page is static, so this script runs server-side in GitHub Actions.
It keeps API keys out of the browser and writes docs/talent-flows.json for the
front-end bubble graph.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "talent-flows.json"
MIRROR_DIRS = [ROOT / "agi", ROOT / "ai-global-index"]
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
USER_AGENT = "IsItHub/1.0 (talent-flow; https://isithub.com/)"

MAX_RESULTS = max(1, min(10, int(os.environ.get("TALENT_FLOW_MAX_RESULTS", "5"))))
MAX_QUERIES = max(0, int(os.environ.get("TALENT_FLOW_MAX_QUERIES", "40")))

COMPANIES = {
    "spacex": {
        "name": "SpaceX",
        "aliases": ["SpaceX", "Starlink"],
        "x": 12,
        "y": 18,
        "zh": "航天工程 / Starlink / 自动化制造",
        "en": "Space systems / Starlink / automated manufacturing",
    },
    "nvidia": {
        "name": "NVIDIA",
        "aliases": ["NVIDIA", "Nvidia", "NVDA"],
        "x": 72,
        "y": 18,
        "zh": "GPU / CUDA / AI 基础设施",
        "en": "GPU / CUDA / AI infrastructure",
    },
    "openai": {
        "name": "OpenAI",
        "aliases": ["OpenAI"],
        "x": 18,
        "y": 62,
        "zh": "前沿模型 / Agent / 产品化",
        "en": "Frontier models / agents / productization",
    },
    "anthropic": {
        "name": "Anthropic",
        "aliases": ["Anthropic", "Claude"],
        "x": 43,
        "y": 68,
        "zh": "Claude / 安全对齐 / 企业模型",
        "en": "Claude / safety alignment / enterprise models",
    },
    "google": {
        "name": "Google",
        "aliases": ["Google", "DeepMind", "Google DeepMind", "Google Brain"],
        "x": 72,
        "y": 62,
        "zh": "DeepMind / TPU / 搜索与云 AI",
        "en": "DeepMind / TPU / search and cloud AI",
    },
    "external": {
        "name": "External Talent Pool",
        "aliases": ["external talent", "startup", "university", "research lab"],
        "x": 43,
        "y": 39,
        "zh": "外部公司 / 高校 / 研究机构",
        "en": "Outside companies / universities / research labs",
    },
}

CORE_COMPANY_KEYS = ["spacex", "nvidia", "openai", "anthropic", "google"]


def quoted_aliases(key: str) -> str:
    return " OR ".join(f'"{alias}"' for alias in COMPANIES[key]["aliases"])


def primary_alias(key: str) -> str:
    return str(COMPANIES[key]["aliases"][0])


def build_talent_queries() -> list[tuple[str, str, str]]:
    queries: list[tuple[str, str, str]] = []
    for from_key in CORE_COMPANY_KEYS:
        for to_key in CORE_COMPANY_KEYS:
            if from_key == to_key:
                continue
            from_terms = quoted_aliases(from_key)
            to_terms = quoted_aliases(to_key)
            queries.append((
                from_key,
                to_key,
                f'({to_terms}) ("joined from" OR "hired from" OR "recruited from" OR "former" OR "alumni") ({from_terms}) ("AI researcher" OR engineer OR executive OR scientist OR talent OR founder)',
            ))
    for key in CORE_COMPANY_KEYS:
        terms = quoted_aliases(key)
        name = primary_alias(key)
        queries.append((
            "external",
            key,
            f'({terms}) ("hired" OR "appointed" OR "named" OR "joins" OR "joined" OR "recruited") ("AI researcher" OR engineer OR executive OR scientist OR talent OR founder) -jobs',
        ))
        queries.append((
            key,
            "external",
            f'({terms}) ("left {name}" OR "leaves {name}" OR "departed {name}" OR "resigned from {name}" OR "former {name}") ("AI researcher" OR engineer OR executive OR scientist OR talent OR founder) -jobs',
        ))
    return queries


PAIR_QUERIES = build_talent_queries()

FALLBACK_EVENTS = [
    {
        "from": "google",
        "to": "openai",
        "title": "DeepMind / Google Brain alumni remain a key talent pool for frontier-model labs",
        "url": "https://isithub.com/",
        "source": "IsItHub seed model",
        "publishedAt": "",
        "people": [],
        "confidence": "seed",
    },
    {
        "from": "nvidia",
        "to": "openai",
        "title": "GPU infrastructure and compiler talent connects NVIDIA to frontier-model deployment",
        "url": "https://isithub.com/",
        "source": "IsItHub seed model",
        "publishedAt": "",
        "people": [],
        "confidence": "seed",
    },
    {
        "from": "openai",
        "to": "spacex",
        "title": "Agent, robotics, and physical-system talent is a crossover watchpoint for SpaceX",
        "url": "https://isithub.com/",
        "source": "IsItHub seed model",
        "publishedAt": "",
        "people": [],
        "confidence": "seed",
    },
]

PERSON_STOPWORDS = {
    "OpenAI", "NVIDIA", "Nvidia", "Google", "DeepMind", "SpaceX", "Starlink",
    "External Talent", "Artificial Intelligence", "Wall Street", "New York", "Silicon Valley",
}


def post_tavily_search(api_key: str, payload: dict[str, object]) -> dict[str, object]:
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
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def source_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        host = ""
    host = host.replace("www.", "")
    if not host:
        return "Tavily"
    parts = host.split(".")
    return parts[-2].title() if len(parts) >= 2 else host.title()


def date_from_result(result: dict[str, object]) -> str:
    raw = str(result.get("published_date") or result.get("publishedAt") or result.get("date") or "")
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        match = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        return f"{match.group(0)}T00:00:00Z" if match else ""


def extract_people(text: str) -> list[str]:
    names = []
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text):
        name = match.group(1).strip()
        if name in PERSON_STOPWORDS:
            continue
        if any(part in PERSON_STOPWORDS for part in name.split()):
            continue
        names.append(name)
    seen = set()
    clean = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            clean.append(name)
    return clean[:4]


def is_talent_signal(text: str) -> bool:
    lower = text.lower()
    signals = [
        "hire", "hired", "joined", "joins", "recruited", "recruits", "poached",
        "former ", "alumni", "left ", "leaves ", "appointed", "named", "talent",
        "researcher", "engineer", "executive", "scientist", "founder",
    ]
    return any(signal in lower for signal in signals)


def parse_result(result: dict[str, object], from_key: str, to_key: str, query: str) -> dict[str, object] | None:
    title = str(result.get("title") or "").strip()
    url = str(result.get("url") or "").strip()
    summary = str(result.get("content") or result.get("summary") or "").strip()
    if not title or not url:
        return None
    text = f"{title} {summary}"
    if not is_talent_signal(text):
        return None
    return {
        "from": from_key,
        "to": to_key,
        "title": title,
        "summary": summary[:280],
        "url": url,
        "source": source_from_url(url),
        "publishedAt": date_from_result(result),
        "people": extract_people(text),
        "query": query,
        "confidence": "tavily",
    }


def dedupe_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    clean = []
    for event in events:
        key = str(event.get("url") or event.get("title") or "").lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        clean.append(event)
    clean.sort(key=lambda item: str(item.get("publishedAt") or ""), reverse=True)
    return clean[:40]


def build_payload(events: list[dict[str, object]], statuses: list[dict[str, object]], source: str) -> dict[str, object]:
    company_map = {key: {**value, "key": key, "inflow": 0, "outflow": 0, "events": 0} for key, value in COMPANIES.items()}
    flow_counts: dict[tuple[str, str], int] = {}
    for event in events:
        from_key = str(event.get("from") or "")
        to_key = str(event.get("to") or "")
        if from_key in company_map:
            company_map[from_key]["outflow"] += 1
            company_map[from_key]["events"] += 1
        if to_key in company_map:
            company_map[to_key]["inflow"] += 1
            company_map[to_key]["events"] += 1
        if from_key and to_key:
            flow_counts[(from_key, to_key)] = flow_counts.get((from_key, to_key), 0) + 1

    flows = []
    for (from_key, to_key), count in sorted(flow_counts.items(), key=lambda item: item[1], reverse=True):
        if from_key not in company_map or to_key not in company_map:
            continue
        matched = [event for event in events if event.get("from") == from_key and event.get("to") == to_key]
        flows.append({
            "from": from_key,
            "to": to_key,
            "count": count,
            "label": f"{company_map[from_key]['name']} → {company_map[to_key]['name']}",
            "events": matched[:6],
        })
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
        "strategy": "Search public news, interviews, speeches, and x.com references for senior AI talent movement among SpaceX, NVIDIA, OpenAI, Anthropic, and Google.",
        "companies": list(company_map.values()),
        "flows": flows,
        "events": events,
        "statuses": statuses,
    }


def main() -> int:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    events: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    if api_key:
        for from_key, to_key, query in PAIR_QUERIES[:MAX_QUERIES]:
            payload = {
                "query": query,
                "topic": "news",
                "search_depth": "basic",
                "max_results": MAX_RESULTS,
                "time_range": "month",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "include_favicon": True,
            }
            try:
                data = post_tavily_search(api_key, payload)
                results = data.get("results") if isinstance(data, dict) else []
                kept = 0
                if isinstance(results, list):
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        event = parse_result(result, from_key, to_key, query)
                        if event:
                            events.append(event)
                            kept += 1
                statuses.append({"query": query, "from": from_key, "to": to_key, "status": "ok", "count": kept})
            except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, OSError, TypeError) as exc:
                statuses.append({"query": query, "from": from_key, "to": to_key, "status": "error", "error": str(exc)[:180]})
    else:
        statuses.append({"status": "skipped", "reason": "TAVILY_API_KEY not set"})

    clean_events = dedupe_events(events)
    source = "Tavily Search"
    if not clean_events:
        clean_events = FALLBACK_EVENTS
        source = "IsItHub seed model"
    payload = build_payload(clean_events, statuses, source)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    OUT.write_text(text, encoding="utf-8")
    for mirror in MIRROR_DIRS:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / OUT.name).write_text(text, encoding="utf-8")
    print(f"wrote {OUT} with {len(payload['companies'])} companies, {len(payload['flows'])} flows, {len(payload['events'])} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
