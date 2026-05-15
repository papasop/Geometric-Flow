#!/usr/bin/env python3
"""Fetch portfolio news into docs/news.json.

Event Registry is the primary unified news layer when EVENT_REGISTRY_API_KEY is
available. Official RSS/Atom sources (SEC, IR, NASA, Defense.gov, WHO, etc.)
are always used as the confirmation/supplement layer.
"""

from __future__ import annotations

import email.utils
import html
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "news.json"
VERSIONED_OUT = ROOT / "news-a8febb0.json"

USER_AGENT = (
    "EntropyAI/1.0 (portfolio-news; https://papasop.github.io/AGI/; "
    "contact: public-site-maintainer)"
)

EVENT_REGISTRY_ENDPOINT = "https://eventregistry.org/api/v1/article/getArticles"
EVENT_REGISTRY_MAX_PER_TOPIC = 18
TRANSLATE_NEWS_TITLES = os.environ.get("TRANSLATE_NEWS_TITLES", "1").strip() != "0"
TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"

OFFICIAL_SOURCES = [
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
    {"name": "WSJ Markets", "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"},
    {"name": "WSJ Business", "url": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"},
    {"name": "Deflation News · TLT Treasuries", "url": "https://news.google.com/rss/search?q=TLT%20OR%20%22long%20Treasury%22%20OR%20%22Treasury%20yields%22%20OR%20%22real%20yields%22%20OR%20%22Fed%20rate%20cuts%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Deflation News · Bitcoin Gold", "url": "https://news.google.com/rss/search?q=Bitcoin%20OR%20BTC%20OR%20GLD%20OR%20%22SPDR%20Gold%20Shares%22%20OR%20%22gold%20prices%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Nikkei Asia", "url": "https://asia.nikkei.com/rss/feed/nar"},
    {"name": "SCMP Tech", "url": "https://www.scmp.com/rss/36/feed"},
    {"name": "SCMP Business", "url": "https://www.scmp.com/rss/92/feed"},
    {"name": "SCMP China", "url": "https://www.scmp.com/rss/4/feed"},
    {"name": "Marketing Dive", "url": "https://www.marketingdive.com/feeds/news/"},
    {"name": "Retail Dive", "url": "https://www.retaildive.com/feeds/news/"},
    {"name": "Food Dive", "url": "https://www.fooddive.com/feeds/news/"},
    {"name": "WWD", "url": "https://wwd.com/feed/"},
    {"name": "MINING.COM", "url": "https://www.mining.com/feed/"},
    {"name": "Mining Technology", "url": "https://www.mining-technology.com/feed/"},
    {"name": "Rare Earth News · Industry", "url": "https://news.google.com/rss/search?q=%22US%20rare%20earth%22%20OR%20%22rare%20earth%20elements%22%20OR%20%22critical%20minerals%22%20OR%20%22neodymium%20magnet%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Rare Earth News · MP", "url": "https://news.google.com/rss/search?q=%22MP%20Materials%22%20OR%20%22Mountain%20Pass%22%20%22rare%20earth%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Rare Earth News · USAR", "url": "https://news.google.com/rss/search?q=%22USA%20Rare%20Earth%22%20OR%20%22USAR%22%20%22Round%20Top%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Rare Earth News · CRML", "url": "https://news.google.com/rss/search?q=%22Critical%20Metals%20Corp%22%20OR%20%22CRML%22%20%22rare%20earth%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "The Robot Report", "url": "https://www.therobotreport.com/feed/"},
    {"name": "Electrek", "url": "https://electrek.co/feed/"},
    {"name": "Automotive World", "url": "https://www.automotiveworld.com/feed/"},
    {"name": "Robotics & Automation News", "url": "https://roboticsandautomationnews.com/feed/"},
    {"name": "Autonomous Vehicle International", "url": "https://www.autonomousvehicleinternational.com/feed"},
    {"name": "SpaceNews", "url": "https://spacenews.com/feed/"},
    {"name": "Defense News", "url": "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml"},
    {"name": "C4ISRNET", "url": "https://www.c4isrnet.com/arc/outboundfeeds/rss/?outputType=xml"},
    {"name": "Breaking Defense", "url": "https://breakingdefense.com/feed/"},
    {"name": "Drone News · Companies", "url": "https://news.google.com/rss/search?q=%22Ondas%22%20OR%20%22Unusual%20Machines%22%20OR%20%22Kopin%22%20OR%20%22Red%20Cat%20Holdings%22%20OR%20%22Kratos%20Defense%22%20OR%20%22Elbit%20Systems%22%20OR%20%22AeroVironment%22%20OR%20%22Terra%20Drone%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Drone News · Tickers", "url": "https://news.google.com/rss/search?q=ONDS%20OR%20UMAC%20OR%20KOPN%20OR%20RCAT%20OR%20KTOS%20OR%20ESLT%20OR%20AVAV%20OR%20%22278A%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Fierce Biotech", "url": "https://www.fiercebiotech.com/rss/biotech/xml"},
    {"name": "BioPharma Dive", "url": "https://www.biopharmadive.com/feeds/news/"},
    {"name": "Endpoints News", "url": "https://endpoints.news/feed/"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
    {"name": "Semiconductor Engineering", "url": "https://semiengineering.com/feed/"},
    {"name": "Palantir IR", "url": "https://investors.palantir.com/news-events/press-releases/rss"},
    {"name": "MP Materials IR", "url": "https://investors.mpmaterials.com/news-releases/news-release-details/rss"},
    {"name": "Apple Newsroom", "url": "https://www.apple.com/newsroom/rss-feed.rss"},
    {"name": "Coca-Cola IR", "url": "https://investors.coca-colacompany.com/news-events/press-releases/rss"},
    {"name": "Brand News · Apple", "url": "https://news.google.com/rss/search?q=%22Apple%22%20AAPL%20iPhone&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Brand News · Coca-Cola", "url": "https://news.google.com/rss/search?q=%22Coca-Cola%22%20KO&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Brand News · NYT", "url": "https://news.google.com/rss/search?q=%22New%20York%20Times%20Company%22%20OR%20%22NYSE%3ANYT%22%20OR%20%22NYT%20stock%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Brand News · Pop Mart", "url": "https://news.google.com/rss/search?q=%22Pop%20Mart%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Brand News · Moutai", "url": "https://news.google.com/rss/search?q=%22Kweichow%20Moutai%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Brand News · JNBY", "url": "https://news.google.com/rss/search?q=%22Jiangnan%20Buyi%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Brand News · PDD", "url": "https://news.google.com/rss/search?q=%22PDD%20Holdings%22%20OR%20Pinduoduo%20OR%20Temu&hl=en-US&gl=US&ceid=US:en"},
    {"name": "China Robotics · Components", "url": "https://news.google.com/rss/search?q=%22Tsugami%20China%22%20OR%20%22%E6%B4%A5%E4%B8%8A%E6%9C%BA%E5%BA%8A%E4%B8%AD%E5%9B%BD%22%20OR%20%2201651.HK%22%20OR%20%22RoboSense%22%20OR%20%22%E9%80%9F%E8%85%BE%E8%81%9A%E5%88%9B%22%20OR%20%2202498.HK%22%20OR%20%22robotics%22%20OR%20%22%E6%9C%BA%E5%99%A8%E4%BA%BA%22&hl=en-US&gl=US&ceid=US:en"},
    {"name": "Huawei Hubble · Components", "url": "https://news.google.com/rss/search?q=%22Huawei%20Hubble%22%20OR%20%22%E5%8D%8E%E4%B8%BA%E5%93%88%E5%8B%83%22%20OR%20%22Tianyue%20Advanced%22%20OR%20%223Peak%22%20OR%20%22JoulWatt%22%20OR%20%22Vanchip%22%20OR%20%22Focuslight%22%20OR%20%22Motorcomm%22%20OR%20%22HHCK%22%20OR%20%22Keli%20Motor%22%20OR%20%22Piotech%22%20OR%20%22Castech%22%20OR%20%22SuperFusion%22%20OR%20%22Semitronix%22%20OR%20%22Yinwang%22%20OR%20%22ModelBest%22%20OR%20%22Qianxun%20Intelligence%22%20OR%20%22GeekSight%22%20OR%20%22%E5%A4%A9%E5%B2%B3%E5%85%88%E8%BF%9B%22%20OR%20%22%E6%80%9D%E7%91%9E%E6%B5%A6%22%20OR%20%22%E4%B8%9C%E8%8A%AF%E8%82%A1%E4%BB%BD%22%20OR%20%22%E6%9D%B0%E5%8D%8E%E7%89%B9%22%20OR%20%22%E5%8D%8E%E6%B5%B7%E8%AF%9A%E7%A7%91%22%20OR%20%22%E7%81%BF%E5%8B%A4%E7%A7%91%E6%8A%80%22%20OR%20%22%E7%82%AC%E5%85%89%E7%A7%91%E6%8A%80%22%20OR%20%22%E8%B5%9B%E7%9B%AE%E7%A7%91%E6%8A%80%22%20OR%20%22%E7%A7%91%E5%8A%9B%E5%B0%94%22%20OR%20%22%E6%8B%93%E8%8D%86%E7%A7%91%E6%8A%80%22%20OR%20%22%E7%A6%8F%E6%99%B6%E7%A7%91%E6%8A%80%20OR%20%E8%B6%85%E8%81%9A%E5%8F%98%20OR%20%E8%B5%9B%E7%BE%8E%E7%89%B9%20OR%20%E5%BC%95%E6%9C%9B%20OR%20%E9%9D%A2%E5%A3%81%E6%99%BA%E8%83%BD%20OR%20%E5%8D%83%E5%AF%BB%E6%99%BA%E8%83%BD%20OR%20%E6%9E%81%E4%BD%B3%E8%A7%86%E7%95%8C%22&hl=en-US&gl=US&ceid=US:en"},
]

# Backward-compatible alias for older callers/comments.
SOURCES = OFFICIAL_SOURCES

PORTFOLIO_KEYWORDS = {
    "drone": [
        "drone", "uav", "uas", "counter-uas", "counter drone",
        "ondas", "onds", "unusual machines", "umac", "kopin", "kopn", "red cat", "rcat",
        "kratos", "ktos", "aerovironment", "avav", "iron beam", "elbit", "eslt", "skydio",
        "terra drone", "278a", "japan drone", "utm", "unmanned traffic management",
    ],
    "satellite-intel": [
        "satellite", "space", "geospatial", "imagery", "isr", "rf", "signal intelligence",
        "palantir", "pltr", "blacksky", "bksy", "spire", "spir", "rocket lab", "rklb",
        "planet labs", "l3harris", "lhx", "spacex", "unseenlabs", "nasa", "esa",
    ],
    "china-robotics": [
        "robotics", "industrial robot", "humanoid robot", "servo",
        "tsugami china", "01651", "1651.hk", "robosense", "02498", "2498.hk",
        "lidar", "robotic vision", "cnc", "machine tools",
        "中国机器人", "机器人", "工业机器人", "人形机器人",
        "津上机床中国", "津上机床", "速腾聚创", "激光雷达", "机器人视觉", "数控机床",
    ],
    "huawei": [
        "huawei", "hubble", "huawei hubble", "semiconductor", "sic", "silicon carbide",
        "analog chip", "power management", "rf front-end", "optical chip", "ethernet phy",
        "memory chip", "wireless charging", "connector", "ceramic filter", "photonics",
        "icv testing", "intelligent connected vehicle", "automotive simulation",
        "688234", "688536", "688110", "688182", "688167", "688261",
        "688048", "688153", "688498", "688141", "688515", "688458", "688629",
        "688535", "02257", "002892", "688072", "002222",
        "superfusion", "semitronix", "yinwang", "modelbest", "qianxun intelligence", "geeksight",
        "华为", "哈勃", "华为哈勃", "半导体", "碳化硅", "模拟芯片", "电源管理",
        "射频前端", "光芯片", "以太网物理层", "存储芯片", "无线充电", "连接器",
        "智能电机", "电机驱动", "半导体设备", "薄膜沉积", "沉积设备",
        "半导体封装材料", "封装材料", "算力", "工业软件", "大模型", "具身智能", "车BU", "智能汽车零部件",
        "非线性光学晶体", "激光光学", "光学元件",
        "陶瓷滤波器", "微波介质陶瓷", "光子技术", "智能网联汽车", "仿真测试",
        "天岳先进", "思瑞浦", "东芯半导体", "东芯股份", "灿勤科技", "炬光科技",
        "东微半导", "长光华芯", "唯捷创芯", "源杰科技", "杰华特", "裕太微", "美芯晟",
        "华丰科技", "华海诚科", "赛目科技", "科力尔", "拓荆科技", "福晶科技",
        "超聚变", "超聚变数字技术", "赛美特", "引望", "深圳引望", "面壁智能", "千寻智能", "极佳视界",
    ],
    "biotech": [
        "biotech", "biotechnology", "pharma", "drug", "therapy", "clinical", "fda",
        "antibody", "gene", "rna", "vaccine", "oncology", "baker bros", "abcellera",
    ],
    "us-rare-earth": [
        "rare earth", "rare-earth", "rare earth elements", "critical minerals",
        "critical metals", "neodymium", "magnet", "permanent magnet",
        "domestic mineral", "mineral supply chain", "mountain pass", "round top",
        "mp materials", "usa rare earth", "usar", "critical metals corp", "crml",
    ],
    "china-property": [
        "hong kong", "property", "real estate", "utilities", "hsbc", "standard chartered",
        "clp", "towngas", "ck hutchison", "sun hung kai",
    ],
    "no-yield": [
        "tlt", "treasury", "treasury bond", "deflation", "bitcoin", "btc", "gld",
        "spdr gold shares", "real yield", "federal reserve", "rate cut", "rate cuts",
    ],
    "brand": [
        "kweichow moutai", "moutai", "600519", "贵州茅台", "茅台",
        "apple", "aapl", "iphone", "app store", "apple tv", "apple music",
        "coca-cola", "coca cola", "coke", "diet coke", "sprite", "fanta", "ko",
        "new york times", "nyt", "nyt cooking", "the athletic",
        "pop mart", "popmart", "9992", "泡泡玛特",
        "jnby", "jiangnan buyi", "3306", "江南布衣",
        "pdd holdings", "pdd", "pinduoduo", "temu", "拼多多",
    ],
}

BRAND_EXCLUDE_KEYWORDS = [
    "chip", "chips", "semiconductor", "semiconductors", "foundry", "wafer",
    "tsmc", "nvidia", "gpu", "ai accelerator", "advanced packaging",
]

SOURCE_REQUIRED_KEYWORDS = {
    "Brand News · NYT": [
        "new york times company", "nyse:nyt", "nyt stock", "nyt shares",
        "the new york times company", "valuation", "earnings", "subscription",
        "subscribers", "revenue",
    ],
    "Rare Earth News · Industry": [
        "us rare earth", "u.s. rare earth", "united states rare earth",
        "american rare earth", "us critical minerals", "u.s. critical minerals",
        "united states critical minerals", "usmca", "mp materials",
        "mountain pass", "usa rare earth", "round top", "critical metals corp", "crml",
    ],
    "Rare Earth News · MP": [
        "mp materials", "mountain pass", "rare earth", "magnet", "defense department",
    ],
    "Rare Earth News · USAR": [
        "usa rare earth", "usar", "round top", "rare earth", "magnet",
    ],
    "Rare Earth News · CRML": [
        "critical metals corp", "crml", "rare earth", "critical minerals", "critical metals",
    ],
}

EXCLUDED_NEWS_KEYWORDS = [
    "yahoo finance", " - yahoo",
    "temu lex luthor", "from temu", "looks like a bmw from temu",
    "temu benson boone", "temu range rover",
]

SOURCE_PORTFOLIO_MATCHES = {
    "Deflation News · TLT Treasuries": ["no-yield"],
    "Deflation News · Bitcoin Gold": ["no-yield"],
    "Apple Newsroom": ["brand"],
    "Coca-Cola IR": ["brand"],
    "Brand News · Apple": ["brand"],
    "Brand News · Coca-Cola": ["brand"],
    "Brand News · NYT": ["brand"],
    "Brand News · Pop Mart": ["brand"],
    "Brand News · Moutai": ["brand"],
    "Brand News · JNBY": ["brand"],
    "Brand News · PDD": ["brand"],
    "China Robotics · Components": ["china-robotics"],
    "Huawei Hubble · Components": ["huawei"],
    "MP Materials IR": ["us-rare-earth"],
    "Rare Earth News · Industry": ["us-rare-earth"],
    "Rare Earth News · MP": ["us-rare-earth"],
    "Rare Earth News · USAR": ["us-rare-earth"],
    "Rare Earth News · CRML": ["us-rare-earth"],
}

NEWS_OVERRIDES = [
]


def clean_text(value: str | None, limit: int = 320) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit].rstrip()


def has_cjk(value: str | None) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value or ""))


def translate_title_to_zh(title: str) -> str:
    if not TRANSLATE_NEWS_TITLES or not title or has_cjk(title):
        return title if has_cjk(title) else ""
    query = urllib.parse.urlencode({
        "client": "gtx",
        "sl": "auto",
        "tl": "zh-CN",
        "dt": "t",
        "q": title,
    })
    req = urllib.request.Request(
        f"{TRANSLATE_ENDPOINT}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        translated = "".join(part[0] for part in payload[0] if part and part[0])
        return clean_text(translated, 180)
    except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, OSError, TypeError, IndexError):
        return ""


def enrich_title_fields(item: dict[str, object]) -> dict[str, object]:
    title = clean_text(str(item.get("title") or ""), 180)
    if not title:
        return item
    item["title"] = title
    if has_cjk(title):
        item.setdefault("titleZh", title)
    else:
        item.setdefault("titleEn", title)
        translated = translate_title_to_zh(title)
        if translated:
            item.setdefault("titleZh", translated)
    return item


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


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


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


def author_of(node: ET.Element) -> str:
    direct = text_of(node, [
        "author",
        "creator",
        "source",
        "{http://purl.org/dc/elements/1.1/}creator",
        "{http://purl.org/dc/terms/}creator",
    ])
    if direct:
        return clean_text(direct, 80)
    for child in node:
        if child.tag.rsplit("}", 1)[-1].lower() != "author":
            continue
        for grandchild in child:
            if grandchild.tag.rsplit("}", 1)[-1].lower() == "name" and grandchild.text:
                return clean_text(grandchild.text, 80)
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
            "author": source["name"],
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
        author = author_of(node) or source["name"]
        published = parse_date(text_of(node, ["pubDate", "published", "updated", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated"]))
        if title and url:
            items.append({
                "source": source["name"],
                "title": title,
                "summary": summary or title,
                "url": url,
                "author": author,
                "publishedAt": published,
            })
    return items


def event_registry_query_for(portfolio: str, keywords: list[str]) -> str:
    # Keep the query focused. Broad one-letter tickers and ultra-generic terms
    # create noisy cross-portfolio matches, so skip them at the API layer and
    # let the local classifier do final matching.
    terms = []
    for keyword in keywords:
        cleaned = re.sub(r"\s+", " ", keyword.strip())
        if len(cleaned) < 3:
            continue
        if cleaned.lower() in {"ai", "rf", "ev", "us", "cn", "hk"}:
            continue
        terms.append(cleaned)
    preferred = terms[:10]
    return " OR ".join(f'"{term}"' if " " in term else term for term in preferred) or portfolio


def parse_event_registry_article(article: dict[str, object], portfolio: str) -> dict[str, object] | None:
    title = clean_text(str(article.get("title") or ""), 180)
    url = str(article.get("url") or "").strip()
    if not title or not url:
        return None
    source = article.get("source")
    source_name = ""
    if isinstance(source, dict):
        source_name = clean_text(str(source.get("title") or source.get("uri") or ""), 80)
    authors = article.get("authors")
    author = ""
    if isinstance(authors, list) and authors:
        first_author = authors[0]
        if isinstance(first_author, dict):
            author = clean_text(str(first_author.get("name") or ""), 80)
        else:
            author = clean_text(str(first_author), 80)
    body = clean_text(str(article.get("body") or article.get("summary") or ""), 320)
    published = parse_date(str(article.get("dateTimePub") or article.get("dateTime") or article.get("date") or ""))
    return {
        "source": source_name or "Event Registry",
        "via": "Event Registry",
        "title": title,
        "summary": body or title,
        "url": url,
        "author": author or source_name or "Event Registry",
        "publishedAt": published,
        "matchedPortfolios": [portfolio],
    }


def fetch_event_registry(api_key: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    items: list[dict[str, object]] = []
    source_status: list[dict[str, object]] = []
    for portfolio, keywords in PORTFOLIO_KEYWORDS.items():
        query = event_registry_query_for(portfolio, keywords)
        payload = {
            "apiKey": api_key,
            "action": "getArticles",
            "resultType": "articles",
            "keyword": query,
            "lang": ["eng", "zho"],
            "articlesSortBy": "date",
            "articlesCount": EVENT_REGISTRY_MAX_PER_TOPIC,
            "articlesIncludeArticleConcepts": True,
            "articlesIncludeArticleCategories": True,
            "articlesIncludeArticleImage": False,
        }
        try:
            data = post_json(EVENT_REGISTRY_ENDPOINT, payload)
            results = (((data.get("articles") or {}) if isinstance(data, dict) else {}).get("results") or [])
            kept = 0
            for article in results:
                if not isinstance(article, dict):
                    continue
                parsed = parse_event_registry_article(article, portfolio)
                if not parsed:
                    continue
                enrich_title_fields(parsed)
                matched, tags = classify(parsed)
                if portfolio not in matched:
                    continue
                parsed["matchedPortfolios"] = matched
                parsed["tags"] = tags
                apply_news_overrides(parsed)
                items.append(parsed)
                kept += 1
            source_status.append({
                "name": "Event Registry",
                "portfolio": portfolio,
                "query": query,
                "status": "ok",
                "count": kept,
            })
            time.sleep(0.25)
        except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, OSError) as exc:
            source_status.append({
                "name": "Event Registry",
                "portfolio": portfolio,
                "query": query,
                "status": "error",
                "error": str(exc)[:180],
                "count": 0,
            })
    return items, source_status


def classify(item: dict[str, str]) -> tuple[list[str], list[str]]:
    text = " ".join([item.get("source", ""), item.get("title", ""), item.get("summary", "")]).lower()
    if any(keyword in text for keyword in EXCLUDED_NEWS_KEYWORDS):
        return [], []
    required = SOURCE_REQUIRED_KEYWORDS.get(item.get("source", ""))
    if required and not any(keyword in text for keyword in required):
        return [], []
    matched = list(SOURCE_PORTFOLIO_MATCHES.get(item.get("source", ""), []))
    tags = []
    if "brand" in matched and any(keyword in text for keyword in BRAND_EXCLUDE_KEYWORDS):
        matched = [portfolio for portfolio in matched if portfolio != "brand"]
    if matched:
        tags.append(item.get("source", "industry source"))
    for portfolio, keywords in PORTFOLIO_KEYWORDS.items():
        if portfolio == "brand" and any(keyword in text for keyword in BRAND_EXCLUDE_KEYWORDS):
            continue
        hits = []
        for keyword in keywords:
            keyword_l = keyword.lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(keyword_l)}(?![a-z0-9])", text):
                hits.append(keyword)
        if hits:
            matched.append(portfolio)
            tags.extend(hits[:3])
    return sorted(set(matched)), sorted(set(tags))[:10]


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

    event_registry_key = os.environ.get("EVENT_REGISTRY_API_KEY", "").strip()
    if event_registry_key:
        event_items, event_status = fetch_event_registry(event_registry_key)
        source_status.extend(event_status)
        for item in event_items:
            key = (item.get("url") or item.get("title") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
    else:
        source_status.append({
            "name": "Event Registry",
            "status": "skipped",
            "count": 0,
            "note": "Set EVENT_REGISTRY_API_KEY to enable the unified primary news layer.",
        })

    for source in OFFICIAL_SOURCES:
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
                enrich_title_fields(item)
                item["matchedPortfolios"] = matched
                item["tags"] = tags
                apply_news_overrides(item)
                items.append(item)
                kept += 1
            source_status.append({"name": source["name"], "url": source["url"], "status": "ok", "count": kept})
            time.sleep(0.35)
        except (urllib.error.URLError, TimeoutError, socket.timeout, ET.ParseError, ValueError, OSError) as exc:
            source_status.append({"name": source["name"], "url": source["url"], "status": "error", "error": str(exc)[:180], "count": 0})

    items.sort(key=lambda item: (bool(item.get("matchedPortfolios")), item.get("publishedAt") or ""), reverse=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "strategy": {
            "primary": "Event Registry",
            "primaryEnabled": bool(event_registry_key),
            "supplements": [
                "SEC", "Company IR", "NASA", "Defense.gov", "WHO",
                "Reuters", "AP", "BBC", "CNBC", "MarketWatch",
                "WSJ / Dow Jones", "Nikkei Asia", "SCMP",
                "Marketing Dive", "Retail Dive", "Food Dive", "WWD",
                "SpaceNews", "Defense News", "C4ISRNET", "Breaking Defense",
                "Fierce Biotech", "BioPharma Dive", "Endpoints News",
                "TechCrunch",
                "Semiconductor Engineering",
            ],
        },
        "sources": source_status,
        "items": items[:200],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    VERSIONED_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} and {VERSIONED_OUT} with {len(payload['items'])} items from {sum(1 for s in source_status if s['status'] == 'ok')} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
