#!/usr/bin/env python3
"""Fetch portfolio news from public RSS/Atom sources into docs/news.json."""

from __future__ import annotations

import email.utils
import html
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "news.json"

USER_AGENT = (
    "EntropyAI/1.0 (portfolio-news; https://papasop.github.io/AGI/; "
    "contact: public-site-maintainer)"
)

SOURCES = [
    {"name": "Reuters", "url": "https://www.reuters.com/arc/outboundfeeds/rss/?outputType=xml"},
    {"name": "AP", "url": "https://apnews.com/hub/ap-top-news?output=rss"},
    {"name": "BBC", "url": "https://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "CNBC", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
    {"name": "SEC filings", "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start=0&count=40&output=atom"},
    {"name": "SEC press", "url": "https://www.sec.gov/news/pressreleases.rss"},
    {"name": "Defense.gov", "url": "https://www.defense.gov/DesktopModules/ArticleCS/RSS.ashx?ContentType=1&Site=945&Category=675"},
    {"name": "NASA", "url": "https://www.nasa.gov/news-release/feed/"},
    {"name": "ESA", "url": "https://www.esa.int/rssfeed/TopNews"},
    {"name": "WHO", "url": "https://www.who.int/rss-feeds/news-english.xml"},
    {"name": "Nasdaq", "url": "https://www.nasdaq.com/feed/rssoutbound?category=Company%20News"},
    {"name": "NYSE", "url": "https://www.nyse.com/rss/news"},
    {"name": "Palantir IR", "url": "https://investors.palantir.com/news-events/press-releases/rss"},
    {"name": "CrowdStrike IR", "url": "https://ir.crowdstrike.com/news-releases/rss"},
    {"name": "MP Materials IR", "url": "https://investors.mpmaterials.com/news-releases/news-release-details/rss"},
]

PORTFOLIO_KEYWORDS = {
    "drone": [
        "drone", "uav", "uas", "counter-uas", "counter drone", "autonomous", "defense",
        "ondas", "onds", "unusual machines", "umac", "kopin", "kopn", "red cat", "rcat",
        "kratos", "ktos", "aerovironment", "avav", "iron beam", "elbit", "eslt", "skydio",
    ],
    "satellite-intel": [
        "satellite", "space", "geospatial", "imagery", "isr", "rf", "signal intelligence",
        "palantir", "pltr", "blacksky", "bksy", "spire", "spir", "rocket lab", "rklb",
        "planet labs", "l3harris", "lhx", "spacex", "unseenlabs", "nasa", "esa",
    ],
    "ai-software": [
        "saas", "software", "agent", "ai", "cloud", "salesforce", "crm", "snowflake", "snow",
        "mongodb", "mdb", "twilio", "twlo", "zoom", "adobe", "adbe", "workday", "wday",
        "google", "goog", "shopify", "shop", "sofi", "hubspot", "hubs", "braze", "brze",
    ],
    "cybersecurity": [
        "cyber", "security", "zero trust", "endpoint", "firewall", "crowdstrike", "crwd",
        "palo alto", "panw", "fortinet", "ftnt", "cloudflare", "net", "zscaler", "zs",
        "datadog", "ddog", "synopsys", "snps", "cadence", "cdns",
    ],
    "china-leading": [
        "pdd", "pinduoduo", "temu", "maifushi", "maifu", "2556", "china", "e-commerce",
        "marketing automation", "midea", "byd",
    ],
    "biotech": [
        "biotech", "biotechnology", "pharma", "drug", "therapy", "clinical", "fda",
        "antibody", "gene", "rna", "vaccine", "oncology", "baker bros", "abcellera",
    ],
    "hantavirus": [
        "hantavirus", "hemorrhagic fever", "outbreak", "who", "moderna", "diagnostic",
        "pcr", "vaccine", "public health",
    ],
    "battery": [
        "battery", "lithium", "energy storage", "catl", "byd", "ev", "electric vehicle",
        "grid storage", "sodium-ion",
    ],
    "us-rare-earth": [
        "rare earth", "critical minerals", "mp materials", "usa rare earth", "usar", "crml",
        "department of defense", "dod", "magnet",
    ],
    "china-property": [
        "hong kong", "property", "real estate", "utilities", "hsbc", "standard chartered",
        "clp", "towngas", "ck hutchison", "sun hung kai",
    ],
    "no-yield": [
        "tlt", "treasury", "yield", "deflation", "bitcoin", "btc", "gold", "gld",
        "rates", "federal reserve",
    ],
    "brand": [
        "brand", "apple", "aapl", "coca-cola", "ko", "new york times", "nyt", "pop mart",
        "moutai", "jnby",
    ],
}

NEWS_OVERRIDES = [
    {
        "title_contains": "Big Tech’s AI spending is depriving investors of juicy payouts",
        "author": "Bill Peters",
        "firstPublishedAt": "2026-05-10T14:00:00Z",
        "updatedAt": "2026-05-10T20:13:00Z",
        "matchedPortfolios": ["ai-software", "no-yield"],
        "tags": ["AI capex", "buybacks", "Goldman Sachs", "S&P 500"],
    },
]


def clean_text(value: str | None, limit: int = 320) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit].rstrip()


def parse_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    return None


def fetch_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*"})
    with urllib.request.urlopen(req, timeout=8) as response:
        return response.read()


def text_of(node: ET.Element, names: list[str]) -> str:
    for name in names:
        found = node.find(name)
        if found is not None and found.text:
            return found.text
    for child in node:
        bare = child.tag.rsplit("}", 1)[-1].lower()
        if bare in {n.rsplit("}", 1)[-1].lower() for n in names} and child.text:
            return child.text
    return ""


def link_of(node: ET.Element) -> str:
    direct = text_of(node, ["link", "{http://www.w3.org/2005/Atom}link"])
    if direct:
        return direct.strip()
    for child in node:
        if child.tag.rsplit("}", 1)[-1].lower() == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
    return ""


def parse_html_page(source: dict[str, str], data: bytes) -> list[dict[str, str]]:
    page = data.decode("utf-8", errors="ignore")
    items = []
    seen = set()
    for match in re.finditer(r'<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', page, re.I | re.S):
        href = html.unescape(match.group(1))
        body = clean_text(match.group(2), 220)
        if len(body) < 18:
            continue
        if not re.search(r"/(article|news|press|release|story|business|technology)/|news-release", href, re.I):
            continue
        url = urljoin(source["url"], href)
        key = (body.lower(), url.lower())
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "source": source["name"],
            "title": body,
            "summary": body,
            "url": url,
            "publishedAt": None,
        })
        if len(items) >= 18:
            break
    return items


def parse_feed(source: dict[str, str], data: bytes) -> list[dict[str, str]]:
    if re.search(br"<html\b|<!doctype html", data[:1000], re.I):
        return parse_html_page(source, data)
    root = ET.fromstring(data)
    root_name = root.tag.rsplit("}", 1)[-1].lower()
    nodes = []
    if root_name == "rss":
        nodes = root.findall("./channel/item")
    elif root_name == "feed":
        nodes = root.findall("{http://www.w3.org/2005/Atom}entry") or root.findall("entry")
    else:
        nodes = root.findall(".//item")

    items = []
    for node in nodes[:24]:
        title = clean_text(text_of(node, ["title", "{http://www.w3.org/2005/Atom}title"]), 180)
        summary = clean_text(text_of(node, ["description", "summary", "content", "{http://www.w3.org/2005/Atom}summary", "{http://purl.org/rss/1.0/modules/content/}encoded"]))
        url = link_of(node)
        published = parse_date(text_of(node, ["pubDate", "published", "updated", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated"]))
        if title and url:
            items.append({
                "source": source["name"],
                "title": title,
                "summary": summary or title,
                "url": url,
                "publishedAt": published,
            })
    return items


def classify(item: dict[str, str]) -> tuple[list[str], list[str]]:
    text = " ".join([item.get("source", ""), item.get("title", ""), item.get("summary", "")]).lower()
    matched = []
    tags = []
    for portfolio, keywords in PORTFOLIO_KEYWORDS.items():
        hits = []
        for keyword in keywords:
            keyword_l = keyword.lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword_l)}(?![a-z0-9])", text):
                hits.append(keyword)
        if hits:
            matched.append(portfolio)
            tags.extend(hits[:3])
    return matched, sorted(set(tags))[:10]


def apply_news_overrides(item: dict[str, object]) -> None:
    title = str(item.get("title") or "")
    for override in NEWS_OVERRIDES:
        if override["title_contains"] not in title:
            continue
        for key, value in override.items():
            if key == "title_contains":
                continue
            if key == "matchedPortfolios":
                existing = item.get("matchedPortfolios") or []
                item[key] = sorted(set([*existing, *value]))
            elif key == "tags":
                existing = item.get("tags") or []
                item[key] = sorted(set([*existing, *value]))[:10]
            else:
                item[key] = value


def main() -> int:
    seen = set()
    items: list[dict[str, object]] = []
    source_status = []

    for source in SOURCES:
        try:
            raw = fetch_url(source["url"])
            parsed = parse_feed(source, raw)
            kept = 0
            for item in parsed:
                key = (item.get("url") or item.get("title") or "").lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                matched, tags = classify(item)
                item["matchedPortfolios"] = matched
                item["tags"] = tags
                apply_news_overrides(item)
                items.append(item)
                kept += 1
            source_status.append({"name": source["name"], "url": source["url"], "status": "ok", "count": kept})
            time.sleep(0.35)
        except (urllib.error.URLError, TimeoutError, socket.timeout, ET.ParseError, ValueError, OSError) as exc:
            source_status.append({"name": source["name"], "url": source["url"], "status": "error", "error": str(exc)[:180], "count": 0})

    items.sort(key=lambda item: item.get("publishedAt") or "", reverse=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources": source_status,
        "items": items[:200],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(payload['items'])} items from {sum(1 for s in source_status if s['status'] == 'ok')} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
