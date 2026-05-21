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
MIRROR_DIRS = [ROOT / "agi", ROOT / "ai-global-index"]

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
    {"name": "AI Cloud · Components", "url": "https://news.google.com/rss/search?q=%22Nebius%22%20OR%20%22NBIS%22%20OR%20%22VNET%22%20OR%20%22%E4%B8%96%E7%BA%AA%E4%BA%92%E8%BF%9E%22%20OR%20%22AI%20cloud%22%20OR%20%22AI%20data%20center%22&hl=en-US&gl=US&ceid=US:en"},
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
    "ai-cloud": [
        "ai cloud", "ai infrastructure", "data center", "ai data center",
        "nebius", "nebius group", "nbis", "vnet", "vnet group", "世纪互连",
        "AI 云", "AI 基础设施", "数据中心", "Nebius", "VNET", "世纪互连", "新云", "AI 新云",
    ],
    "huawei": [
        "huawei", "hubble", "huawei hubble", "semiconductor", "sic", "silicon carbide",
        "analog chip", "power management", "rf front-end", "optical chip", "ethernet phy",
        "memory chip", "wireless charging", "connector", "ceramic filter", "photonics",
        "icv testing", "intelligent connected vehicle", "automotive simulation",
        "hestia power", "qiangyi", "onmicro", "tysic", "sidea", "probe testing", "probe station",
        "688234", "688536", "688110", "688182", "688167", "688261",
        "688048", "688153", "688498", "688141", "688515", "688458", "688629",
        "688535", "02726", "688809", "688790", "02658", "02257", "002892", "688072", "002222", "301629", "06656",
        "superfusion", "semitronix", "yinwang", "modelbest", "qianxun intelligence", "geeksight",
        "华为", "哈勃", "华为哈勃", "半导体", "碳化硅", "模拟芯片", "电源管理",
        "射频前端", "光芯片", "以太网物理层", "存储芯片", "无线充电", "连接器",
        "智能电机", "电机驱动", "半导体设备", "薄膜沉积", "沉积设备",
        "半导体封装材料", "封装材料", "算力", "工业软件", "大模型", "具身智能", "车BU", "智能汽车零部件",
        "非线性光学晶体", "激光光学", "光学元件", "探针测试", "探针台",
        "陶瓷滤波器", "微波介质陶瓷", "光子技术", "智能网联汽车", "仿真测试",
        "天岳先进", "思瑞浦", "东芯半导体", "东芯股份", "瀚天天成", "灿勤科技", "炬光科技",
        "东微半导", "长光华芯", "唯捷创芯", "源杰科技", "杰华特", "裕太微", "美芯晟",
        "强一股份", "昂瑞微", "华丰科技", "华海诚科", "天域半导体", "赛目科技", "科力尔", "拓荆科技", "福晶科技", "矽电股份", "思格新能源",
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
    "AI Cloud · Components": ["ai-cloud"],
    "Huawei Hubble · Components": ["huawei"],
    "MP Materials IR": ["us-rare-earth"],
    "Rare Earth News · Industry": ["us-rare-earth"],
    "Rare Earth News · MP": ["us-rare-earth"],
    "Rare Earth News · USAR": ["us-rare-earth"],
    "Rare Earth News · CRML": ["us-rare-earth"],
}

AI_CLOUD_NEWS_COMPANIES = [
    {
        "source": "AI Cloud · Nebius Group",
        "terms": ["Nebius Group", "Nebius", "NBIS"],
    },
    {
        "source": "AI Cloud · VNET",
        "terms": ["VNET", "VNET Group", "\u4e16\u7eaa\u4e92\u8fde"],
    },
]

def google_news_source(name: str, terms: list[str]) -> dict[str, str]:
    query = " OR ".join(f'"{term}"' for term in terms)
    return {
        "name": name,
        "url": (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        ),
    }


def google_news_query_source(name: str, query: str) -> dict[str, str]:
    return {
        "name": name,
        "url": (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        ),
    }


def extract_company_news_terms() -> list[str]:
    """Use the site ranking table as the company-news keyword source."""
    html_path = ROOT / "index.html"
    try:
        html_text = html_path.read_text(encoding="utf-8")
    except OSError:
        return []
    terms = []
    for match in re.finditer(r'\[\s*"[^"]+"\s*,\s*"([^"]+)"', html_text):
        name = html.unescape(match.group(1)).strip()
        if len(name) < 2:
            continue
        terms.append(name)
    return sorted(dict.fromkeys(terms), key=str.casefold)


def quoted_or_query(terms: list[str]) -> str:
    return " OR ".join(f'"{term}"' for term in terms)


def chunked_terms(terms: list[str], size: int = 18) -> list[list[str]]:
    return [terms[index:index + size] for index in range(0, len(terms), size)]


AI_MARKET_NEWS_QUERY = (
    '("artificial intelligence" OR AI OR Nvidia OR OpenAI OR Anthropic OR '
    '"AI chip" OR semiconductor OR "data center" OR "AI cloud" OR '
    '"large language model" OR "machine learning")'
)
AI_COMPANY_NEWS_TERMS = extract_company_news_terms() or [
    "NVIDIA", "Microsoft", "OpenAI", "Apple", "Alphabet", "Amazon",
    "Anthropic", "Meta Platforms", "Tesla", "SpaceX", "Broadcom", "Palantir",
    "CoreWeave", "Nebius Group", "Tencent", "Alibaba", "Baidu", "寒武纪",
]
AI_PERSON_NEWS_TERMS = [
    "Sam Altman", "Elon Musk", "马斯克", "Jensen Huang", "黄仁勋", "Satya Nadella",
    "Dario Amodei", "Demis Hassabis", "哈萨比斯", "Mark Zuckerberg", "Sundar Pichai",
    "Yann LeCun", "杨立昆", "杨丽坤", "Andrew Ng", "Andrej Karpathy", "Fei-Fei Li",
    "Ilya Sutskever", "Leopold Aschenbrenner", "Lisa Su", "Hock Tan",
    "Masayoshi Son", "Mira Murati", "Kai-Fu Lee", "李飞飞", "李开复",
    "physical world model", "world model", "物理世界模型",
]
AI_SPEECH_PERSON_TERMS = [
    term for term in AI_PERSON_NEWS_TERMS
    if term not in {"physical world model", "world model", "物理世界模型"}
]
AI_PERSON_NEWS_QUERY = (
    f"({quoted_or_query(AI_PERSON_NEWS_TERMS)}) "
    '("AI" OR "artificial intelligence" OR "OpenAI" OR "Nvidia" OR '
    '"large language model" OR "AI chip" OR "agent" OR "robotics" OR '
    '"physical world model" OR "world model" OR "人工智能" OR "物理世界模型") '
    '("says" OR "said" OR "warns" OR "predicts" OR "tweet" OR "post" OR "X" OR "推特" OR "发文" OR "表示")'
)
DEAL_NEWS_TERMS_QUERY = (
    '("equity" OR "stake" OR "financing" OR "funding" OR "financial" OR '
    '"capital" OR "valuation" OR "investment" OR "raises" OR "IPO" OR '
    '"private placement" OR "venture capital" OR "debt financing" OR '
    '"股权" OR "融资" OR "金融" OR "资本" OR "估值" OR "投资")'
)
MNA_NEWS_TERMS_QUERY = (
    '("acquires" OR "acquisition" OR "merger" OR "M&A" OR "buys stake" OR '
    '"takes stake" OR "takes a stake" OR "invests in" OR "investment in" OR '
    '"stake in" OR "to buy" OR "buyout" OR "takeover" OR "收购" OR "并购" OR '
    '"入股" OR "持股" OR "股权")'
)
MNA_REQUIRED_KEYWORDS = [
    "acquires", "acquisition", "merger", "m&a", "buys stake", "takes stake",
    "takes a stake", "invests in", "investment in", "stake in",
    "to buy", "buyout", "takeover", "收购", "并购", "入股", "持股", "股权",
]
DEAL_REQUIRED_KEYWORDS = [
    "equity", "stake", "financing", "funding", "financial", "capital",
    "valuation", "investment", "raises", "raised", "ipo", "private placement",
    "venture capital", "debt financing", "series a", "series b", "series c",
    "backed", "funds", "股权", "融资", "金融", "资本", "估值", "投资",
]

INDUSTRY_NEWS_SOURCES = [
    google_news_query_source("Wall Street Journal", f"site:wsj.com {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("New York Times", f"site:nytimes.com {AI_MARKET_NEWS_QUERY}"),
]
PERSON_NEWS_SOURCES = [
    google_news_query_source("AI Statements · People", AI_PERSON_NEWS_QUERY),
]
COMPANY_NEWS_SOURCES = [
    google_news_query_source(f"Bloomberg · Company Batch {index + 1}", f"site:bloomberg.com ({quoted_or_query(batch)})")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Reuters · Company Batch {index + 1}", f"site:reuters.com ({quoted_or_query(batch)})")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Google · Company Batch {index + 1}", f"({quoted_or_query(batch)}) {AI_MARKET_NEWS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Yahoo · Company Batch {index + 1}", f"(site:yahoo.com OR site:finance.yahoo.com) ({quoted_or_query(batch)}) {AI_MARKET_NEWS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
]
MNA_NEWS_SOURCES = [
    google_news_query_source(f"Bloomberg · M&A Batch {index + 1}", f"site:bloomberg.com ({quoted_or_query(batch)}) {MNA_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Reuters · M&A Batch {index + 1}", f"site:reuters.com ({quoted_or_query(batch)}) {MNA_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Google · M&A Batch {index + 1}", f"({quoted_or_query(batch)}) {MNA_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Yahoo · M&A Batch {index + 1}", f"(site:yahoo.com OR site:finance.yahoo.com) ({quoted_or_query(batch)}) {MNA_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
]
EQUITY_NEWS_SOURCES = [
    google_news_query_source("Bloomberg · Equity Financing", f"site:bloomberg.com {AI_MARKET_NEWS_QUERY} {DEAL_NEWS_TERMS_QUERY}"),
    google_news_query_source("Reuters · Equity Financing", f"site:reuters.com {AI_MARKET_NEWS_QUERY} {DEAL_NEWS_TERMS_QUERY}"),
    google_news_query_source("Google · Equity Financing", f"{AI_MARKET_NEWS_QUERY} {DEAL_NEWS_TERMS_QUERY}"),
    google_news_query_source("Yahoo · Equity Financing", f"(site:yahoo.com OR site:finance.yahoo.com) {AI_MARKET_NEWS_QUERY} {DEAL_NEWS_TERMS_QUERY}"),
] + [
    google_news_query_source(f"Bloomberg · Equity Batch {index + 1}", f"site:bloomberg.com ({quoted_or_query(batch)}) {DEAL_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Reuters · Equity Batch {index + 1}", f"site:reuters.com ({quoted_or_query(batch)}) {DEAL_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Google · Equity Batch {index + 1}", f"({quoted_or_query(batch)}) {DEAL_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
] + [
    google_news_query_source(f"Yahoo · Equity Batch {index + 1}", f"(site:yahoo.com OR site:finance.yahoo.com) ({quoted_or_query(batch)}) {DEAL_NEWS_TERMS_QUERY}")
    for index, batch in enumerate(chunked_terms(AI_COMPANY_NEWS_TERMS))
]
MARKET_NEWS_SOURCES = MNA_NEWS_SOURCES + EQUITY_NEWS_SOURCES
for source in MNA_NEWS_SOURCES:
    source["requiredKeywords"] = MNA_REQUIRED_KEYWORDS
for source in EQUITY_NEWS_SOURCES:
    source["requiredKeywords"] = DEAL_REQUIRED_KEYWORDS
TECH_NEWS_SOURCES = [
    google_news_query_source("MIT Technology Review", f"site:technologyreview.com {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("Wired", f"site:wired.com {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("Stanford Engineering Magazine", f"site:engineering.stanford.edu/magazine {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("Stanford Business Magazine", f"site:gsb.stanford.edu/stanford-business {AI_MARKET_NEWS_QUERY}"),
]
PAPER_NEWS_SOURCES = [
    google_news_query_source("Nature", f"site:nature.com {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("Nature Machine Intelligence", f"site:nature.com/natmachintell {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("Science", f"site:science.org {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("Science Robotics", f"site:science.org/journal/scirobotics {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("NeurIPS", f"site:neurips.cc {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("ICML", f"site:icml.cc {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("ICLR", f"site:openreview.net ICLR {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("ACL Anthology", f"site:aclanthology.org {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("CVF Open Access", f"site:openaccess.thecvf.com {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("AAAI", f"site:ojs.aaai.org {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("JMLR", f"site:jmlr.org {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("PMLR", f"site:proceedings.mlr.press {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("IEEE TPAMI", f"site:ieeexplore.ieee.org TPAMI {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("ACM Digital Library", f"site:dl.acm.org {AI_MARKET_NEWS_QUERY}"),
    google_news_query_source("arXiv AI", f"site:arxiv.org {AI_MARKET_NEWS_QUERY}"),
]
NEWS_SECTIONS = [
    {
        "id": "industry",
        "title": "行业",
        "note": "Wall Street Journal / New York Times",
        "sources": INDUSTRY_NEWS_SOURCES,
    },
    {
        "id": "person",
        "title": "言论",
        "note": "来源：关键词：马斯克 / 黄仁勋 / 哈萨比斯 / 杨立昆 / 杨丽坤 / Sam Altman / Dario Amodei / Fei-Fei Li / 物理世界模型",
        "sources": PERSON_NEWS_SOURCES,
        "derivedFromAll": True,
    },
    {
        "id": "company",
        "title": "公司",
        "note": "Bloomberg / Reuters / Google / Yahoo + M&A / financing keywords",
        "sources": COMPANY_NEWS_SOURCES + MARKET_NEWS_SOURCES,
    },
    {
        "id": "tech",
        "title": "前沿",
        "note": "MIT Technology Review / Wired / Stanford Engineering / Stanford Business",
        "sources": TECH_NEWS_SOURCES,
    },
    {
        "id": "papers",
        "title": "论文",
        "note": "Nature / Science / NeurIPS / ICML / ICLR / ACL / CVPR / AAAI / JMLR / arXiv",
        "sources": PAPER_NEWS_SOURCES,
    },
]
OFFICIAL_SOURCES = [source for section in NEWS_SECTIONS for source in section["sources"]]
SOURCES = OFFICIAL_SOURCES

AI_MARKET_REQUIRED_KEYWORDS = [
    "a.i.", "artificial intelligence", "openai", "anthropic", "chatgpt",
    "nvidia", "ai chip", "ai chips", "semiconductor", "data center",
    "ai cloud", "large language model", "machine learning", "automation",
    "robot", "robotics", "algorithm", "google", "microsoft", "meta",
]
SOURCE_REQUIRED_KEYWORDS.update({
    source["name"]: AI_MARKET_REQUIRED_KEYWORDS for source in OFFICIAL_SOURCES
})
SOURCE_REQUIRED_KEYWORDS.update({
    source["name"]: [
        *[term.lower() for term in AI_PERSON_NEWS_TERMS],
        "ai", "artificial intelligence", "openai", "nvidia",
        "large language model", "人工智能",
    ]
    for source in PERSON_NEWS_SOURCES
})
SOURCE_REQUIRED_KEYWORDS.update({
    source["name"]: [term.lower() for term in AI_COMPANY_NEWS_TERMS]
    for source in COMPANY_NEWS_SOURCES
})
SOURCE_REQUIRED_KEYWORDS.update({
    source["name"]: [
        *[term.lower() for term in AI_COMPANY_NEWS_TERMS],
        *MNA_REQUIRED_KEYWORDS,
        *DEAL_REQUIRED_KEYWORDS,
    ]
    for source in MARKET_NEWS_SOURCES
})

PORTFOLIO_KEYWORDS = {
    "ai-market": sorted(set([
        *PORTFOLIO_KEYWORDS.get("ai-cloud", []),
        *AI_COMPANY_NEWS_TERMS,
        "Nvidia", "OpenAI", "Anthropic", "Microsoft", "Google", "Amazon",
        "Meta", "Broadcom", "Marvell", "TSMC", "AI chip", "AI data center",
        "artificial intelligence", "large language model",
        *(term for company in AI_CLOUD_NEWS_COMPANIES for term in company["terms"]),
    ])),
}

SOURCE_PORTFOLIO_MATCHES = {
    source["name"]: ["ai-market"] for source in OFFICIAL_SOURCES
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


def source_details_of(node: ET.Element) -> tuple[str, str]:
    for child in node:
        if child.tag.rsplit("}", 1)[-1].lower() != "source":
            continue
        name = clean_text(child.text or "", 80)
        url = clean_text(child.attrib.get("url", ""), 240)
        return name, url
    return "", ""


def author_of(node: ET.Element) -> str:
    direct = text_of(node, [
        "author",
        "creator",
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


PUBLISHER_AUTHOR_RE = re.compile(
    r"\b("
    r"reuters|bloomberg|associated press|ap news|cnbc|cnn|bbc|yahoo|google|marketwatch|"
    r"wall street journal|new york times|the information|the verge|techcrunch|wired|"
    r"forbes|fortune|business insider|financial times|investing\.com|seeking alpha|"
    r"the motley fool|tech advisor|coindesk|cointelegraph|coincentral|the globe and mail|"
    r"news|journal|times|post|daily|weekly|magazine|review|press|media"
    r")\b",
    re.I,
)


def looks_like_person_name(author: str) -> bool:
    author = clean_text(author, 80)
    if not author:
        return False
    if re.search(r"[@:/\\]|\d", author):
        return False
    author = re.sub(r"^(by|from)\s+", "", author, flags=re.I).strip()
    author = re.split(r"\s+(?:and|&|,)\s+", author, maxsplit=1, flags=re.I)[0].strip()
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", author):
        return True
    parts = [part for part in re.split(r"\s+", author.replace(".", ". ")) if part]
    if len(parts) < 2 or len(parts) > 4:
        return False
    allowed_particles = {"de", "del", "der", "van", "von", "da", "di", "la", "le", "du"}
    personal_parts = 0
    for part in parts:
        cleaned = part.strip(".,'’()-")
        if not cleaned:
            continue
        if cleaned.lower() in allowed_particles:
            continue
        if re.fullmatch(r"[A-Z]\.?", cleaned):
            personal_parts += 1
            continue
        if re.fullmatch(r"[A-Z][a-z]+(?:[-'’][A-Z][a-z]+)?", cleaned):
            personal_parts += 1
            continue
        return False
    return personal_parts >= 2


def normalize_author(author: str, source_name: str) -> str:
    author = clean_text(author, 80)
    source_name = clean_text(source_name, 80)
    if not author:
        return ""
    author = re.sub(r"^(by|from)\s+", "", author, flags=re.I).strip()
    author = re.split(r"\s+(?:and|&|,)\s+", author, maxsplit=1, flags=re.I)[0].strip()
    if author.lower() == source_name.lower():
        return ""
    if re.search(r"\beditorial\b|\bstaff\b|^news$", author, re.I):
        return ""
    if PUBLISHER_AUTHOR_RE.search(author):
        return ""
    if not looks_like_person_name(author):
        return ""
    return author


def host_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def google_news_article_id(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if "articles" not in parts:
        return ""
    index = parts.index("articles")
    if index + 1 >= len(parts):
        return ""
    return parts[index + 1].strip()


def normalize_url_candidate(url: str) -> str:
    url = html.unescape(url)
    url = url.replace("\\u003d", "=").replace("\\u0026", "&").replace("\\/", "/")
    url = url.strip().strip('"\'.,)]}')
    return url


def resolve_google_news_url(url: str, source_url: str = "") -> str:
    article_id = google_news_article_id(url)
    if not article_id:
        return ""
    try:
        page = fetch_url(url).decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, OSError, UnicodeError):
        return ""

    preferred_host = host_of(source_url)
    raw_urls = re.findall(r'https?://[^"\'<>\s\\]+', page)
    for raw in raw_urls:
        candidate = normalize_url_candidate(raw)
        candidate_host = host_of(candidate)
        if not candidate_host or any(blocked in candidate_host for blocked in ("google.", "gstatic.", "googleusercontent.", "schema.org")):
            continue
        if preferred_host and preferred_host not in candidate_host and candidate_host not in preferred_host:
            continue
        return candidate

    signature_match = re.search(r'data-n-a-sg=["\']([^"\']+)["\']', page)
    timestamp_match = re.search(r'data-n-a-ts=["\'](\d+)["\']', page)
    if not signature_match or not timestamp_match:
        return ""

    timestamp = int(timestamp_match.group(1))
    signature = signature_match.group(1)
    inner = [
        "garturlreq",
        [
            ["en-US", "US", ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"], None, None, 1, 1, "US:en", None, 180, None, None, None, None, None, 0, None, None, [timestamp, 0]],
            "en-US", "US", 1, [2, 3, 4, 8], 1, 0, "655000234", 0, 0, None, 0,
        ],
        article_id,
        timestamp,
        signature,
    ]
    request_body = urllib.parse.urlencode({
        "f.req": json.dumps([[["Fbv4je", json.dumps(inner, separators=(",", ":")), None, "generic"]]], separators=(",", ":")),
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
        data=request_body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=20).read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, OSError, UnicodeError):
        return ""
    for raw in re.findall(r'https?://[^"\\]+', response):
        candidate = normalize_url_candidate(raw)
        candidate_host = host_of(candidate)
        if not candidate_host or "google." in candidate_host:
            continue
        if preferred_host and preferred_host not in candidate_host and candidate_host not in preferred_host:
            continue
        return candidate
    return ""


def extract_article_author(url: str, source_name: str, source_url: str = "") -> str:
    if not url:
        return ""
    if "news.google.com" in url:
        original_url = resolve_google_news_url(url, source_url)
        if not original_url or original_url == url:
            return ""
        url = original_url
    try:
        page = fetch_url(url).decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, socket.timeout, ValueError, OSError, UnicodeError):
        return ""

    candidates: list[str] = []
    meta_patterns = [
        r'<meta[^>]+(?:name|property)=["\'](?:author|article:author|parsely-author|sailthru.author|byl)["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\'](?:author|article:author|parsely-author|sailthru.author|byl)["\']',
    ]
    for pattern in meta_patterns:
        candidates.extend(re.findall(pattern, page, flags=re.I | re.S))

    jsonld_patterns = [
        r'"author"\s*:\s*\{[^{}]*"name"\s*:\s*"([^"]+)"',
        r'"author"\s*:\s*\[[^\]]*?"name"\s*:\s*"([^"]+)"',
        r'"name"\s*:\s*"([^"]+)"\s*,\s*"@type"\s*:\s*"Person"',
    ]
    for pattern in jsonld_patterns:
        candidates.extend(re.findall(pattern, page, flags=re.I | re.S))

    byline_patterns = [
        r'<[^>]+(?:class|id)=["\'][^"\']*(?:byline|author|article-author)[^"\']*["\'][^>]*>(.*?)</[^>]+>',
        r'\bBy\s+([A-Z][A-Za-z.\'’\-]+(?:\s+[A-Z][A-Za-z.\'’\-]+){1,3})\b',
    ]
    for pattern in byline_patterns:
        candidates.extend(re.findall(pattern, page, flags=re.I | re.S))

    for candidate in candidates:
        text = re.sub(r"<[^>]+>", " ", html.unescape(str(candidate)))
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^(by|from)\s+", "", text, flags=re.I).strip()
        author = normalize_author(text, source_name)
        if author:
            return author
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
            "author": "",
            "publishedAt": None,
        })
        if len(items) >= 18:
            break
    return items


def parse_feed(source: dict[str, str], data: bytes) -> list[dict[str, str]]:
    if re.search(br"<html\b|<!doctype html", data[:1000], re.I):
        return parse_html_page(source, data)
    is_google_news_feed = "news.google.com" in source.get("url", "")
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
        rss_source_name, rss_source_url = source_details_of(node)
        display_source = rss_source_name or source["name"]
        author = "" if is_google_news_feed else normalize_author(author_of(node), display_source)
        published = parse_date(text_of(node, ["pubDate", "published", "updated", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated"]))
        if title and url:
            items.append({
                "source": display_source,
                "feedSource": source["name"],
                "sourceUrl": rss_source_url,
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
        "author": normalize_author(author, source_name),
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
    if str(item.get("source") or "").startswith("AI Cloud · "):
        matched.append("ai-cloud")
        tags.append(str(item.get("source") or "AI Cloud"))
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


def item_matches_terms(item: dict[str, str], terms: list[str]) -> bool:
    text = " ".join([item.get("title", ""), item.get("summary", ""), item.get("author", "")]).lower()
    return any(term.lower() in text for term in terms)


SPEECH_SIGNAL_RE = re.compile(
    r"\b(says|said|tells|told|warns|warned|predicts|predicted|argues|argued|"
    r"thinks|believes|expects|calls|called|urges|urged|announces|announced|"
    r"tweeted|posted|wrote|commented|remarks|remarked)\b|"
    r"表示|称|认为|警告|预测|指出|发文|写道|透露|宣布|呼吁|谈到|说道|表示：|“|”|\"",
    re.I,
)
WORLD_MODEL_TERMS = ["physical world model", "world model", "物理世界模型"]
PASSIVE_SPEECH_RE = re.compile(r"\bis\s+said\s+to\b|\bare\s+said\s+to\b|\bwas\s+said\s+to\b|\bwere\s+said\s+to\b", re.I)
REAL_SPEECH_RE = re.compile(
    r"\b(says|said|tells|told|warns|warned|predicts|predicted|argues|argued|"
    r"thinks|believes|expects|calls|called|urges|urged|announces|announced|"
    r"tweeted|posted|wrote|commented|remarks|remarked)\b\s+(?:that\s+)?[A-Za-z0-9'’“\"].{12,}|"
    r"(?:表示|称|认为|警告|预测|指出|发文|写道|透露|宣布|呼吁|谈到|说道)[:：]?\s*.{6,}",
    re.I,
)
QUOTED_STATEMENT_RE = re.compile(r"[\"“‘][^\"”’]{8,}[\"”’]")


def item_matches_speech(item: dict[str, object]) -> bool:
    text = " ".join([
        str(item.get("title") or ""),
        str(item.get("summary") or ""),
        str(item.get("author") or ""),
    ])
    if PASSIVE_SPEECH_RE.search(text):
        return False
    text_l = text.lower()
    has_person = any(re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text_l) for term in AI_SPEECH_PERSON_TERMS)
    has_world_model = any(term.lower() in text_l for term in WORLD_MODEL_TERMS)
    if not (has_person or has_world_model):
        return False
    has_real_statement = bool(REAL_SPEECH_RE.search(text) or QUOTED_STATEMENT_RE.search(text))
    return bool(SPEECH_SIGNAL_RE.search(text) and has_real_statement)


def build_speech_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    speech_items = []
    for item in sorted(items, key=lambda entry: entry.get("publishedAt") or "", reverse=True):
        if not item_matches_speech(item):
            continue
        key = (item.get("url") or item.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        speech_item = dict(item)
        speech_item["section"] = "person"
        speech_item["sectionTitle"] = "言论"
        speech_items.append(speech_item)
        if len(speech_items) >= 32:
            break
    return speech_items


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
    items: list[dict[str, object]] = []
    source_status = []
    section_payload: dict[str, dict[str, object]] = {}

    event_registry_key = os.environ.get("EVENT_REGISTRY_API_KEY", "").strip()
    if event_registry_key:
        event_items, event_status = fetch_event_registry(event_registry_key)
        source_status.extend(event_status)
        for item in event_items:
            key = (item.get("url") or item.get("title") or "").lower()
            if not key:
                continue
            items.append(item)

    for section in NEWS_SECTIONS:
        section_seen = set()
        section_items: list[dict[str, object]] = []
        section_required_keywords = section.get("requiredKeywords", [])
        author_fetches = 0
        for source in section["sources"]:
            source_required_keywords = source.get("requiredKeywords", [])
            try:
                raw = fetch_url(source["url"])
                parsed = parse_feed(source, raw)
                kept = 0
                for item in parsed:
                    key = (item.get("url") or item.get("title") or "").lower()
                    if not key or key in section_seen:
                        continue
                    if section_required_keywords and not item_matches_terms(item, section_required_keywords):
                        continue
                    if source_required_keywords and not item_matches_terms(item, source_required_keywords):
                        continue
                    matched, tags = classify(item)
                    if not matched and not item_matches_terms(item, AI_MARKET_REQUIRED_KEYWORDS):
                        continue
                    if not item.get("author") and author_fetches < 24:
                        item["author"] = extract_article_author(
                            str(item.get("url") or ""),
                            str(item.get("source") or source["name"]),
                            str(item.get("sourceUrl") or ""),
                        )
                        author_fetches += 1
                    section_seen.add(key)
                    enrich_title_fields(item)
                    item["section"] = section["id"]
                    item["sectionTitle"] = section["title"]
                    item["matchedPortfolios"] = matched
                    item["tags"] = tags
                    apply_news_overrides(item)
                    section_items.append(item)
                    items.append(item)
                    kept += 1
                source_status.append({
                    "section": section["id"],
                    "name": source["name"],
                    "url": source["url"],
                    "status": "ok",
                    "count": kept,
                })
                time.sleep(0.35)
            except (urllib.error.URLError, TimeoutError, socket.timeout, ET.ParseError, ValueError, OSError) as exc:
                source_status.append({
                    "section": section["id"],
                    "name": source["name"],
                    "url": source["url"],
                    "status": "error",
                    "error": str(exc)[:180],
                    "count": 0,
                })
        section_items.sort(key=lambda item: item.get("publishedAt") or "", reverse=True)
        section_payload[section["id"]] = {
            "title": section["title"],
            "note": section["note"],
            "items": section_items[:32],
        }

    speech_section = next((section for section in NEWS_SECTIONS if section.get("id") == "person"), None)
    if speech_section:
        existing_speech_items = [
            item for item in section_payload.get("person", {}).get("items", [])
            if item_matches_speech(item)
        ]
        speech_seen = set()
        merged_speech_items = []
        for item in [*existing_speech_items, *build_speech_items(items)]:
            key = (item.get("url") or item.get("title") or "").lower()
            if not key or key in speech_seen:
                continue
            speech_seen.add(key)
            merged_speech_items.append(item)
        merged_speech_items.sort(key=lambda item: item.get("publishedAt") or "", reverse=True)
        section_payload["person"] = {
            "title": speech_section["title"],
            "note": f"{speech_section['note']}；仅保留带真实发言内容的新闻",
            "items": merged_speech_items[:32],
        }

    items.sort(key=lambda item: (bool(item.get("matchedPortfolios")), item.get("publishedAt") or ""), reverse=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "strategy": {
            "primary": "Tabbed news sections: industry from WSJ/NYT, statements derived from all sources by named AI figures and speech signals, company from company-name searches plus M&A and financing keywords, frontier technology from Wired/MIT Technology Review/Stanford sources, papers from top AI journals, conferences, proceedings, and preprint sources",
            "primaryEnabled": True,
            "supplements": [
                "Wall Street Journal",
                "New York Times",
                "Statements",
                "Bloomberg",
                "Reuters",
                "Wired",
                "MIT Technology Review",
                "Stanford Engineering Magazine",
                "Stanford Business Magazine",
                "Nature",
                "Nature Machine Intelligence",
                "Science",
                "Science Robotics",
                "NeurIPS",
                "ICML",
                "ICLR",
                "ACL Anthology",
                "CVF Open Access",
                "AAAI",
                "JMLR",
                "PMLR",
                "IEEE TPAMI",
                "ACM Digital Library",
                "arXiv AI",
            ],
        },
        "sources": source_status,
        "sections": section_payload,
        "items": items[:200],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    OUT.write_text(text, encoding="utf-8")
    VERSIONED_OUT.write_text(text, encoding="utf-8")
    for mirror in MIRROR_DIRS:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / OUT.name).write_text(text, encoding="utf-8")
        (mirror / VERSIONED_OUT.name).write_text(text, encoding="utf-8")
    print(f"wrote {OUT} and {VERSIONED_OUT} with {len(payload['items'])} items from {sum(1 for s in source_status if s['status'] == 'ok')} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
