#!/usr/bin/env python3
"""
Generate docs/quotes.json for the AI Market Cap page.

The static page reads this file first so visitors do not need to hit Yahoo
Finance directly from the browser. Pre-IPO rows use local estimates in the
HTML and are intentionally skipped here.
"""

from __future__ import annotations

import json
import math
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
OUTFILE = ROOT / "quotes.json"
MIRROR_DIRS = [ROOT / "agi", ROOT / "ai-global-index"]
FX_TICKERS = {
    "USDCNY": "CNY=X",
    "USDKRW": "KRW=X",
}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def percent_move(current, previous):
    current = finite(current)
    previous = finite(previous)
    if current is None or previous is None or previous <= 0:
        return None
    return ((current / previous) - 1) * 100


def normalize_spark(values):
    clean = [finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    if len(clean) < 2:
        return None
    low = min(clean)
    high = max(clean)
    if high == low:
        return [17 for _ in clean]
    return [30 - ((value - low) / (high - low)) * 24 for value in clean]


def extract_listed_tickers(html):
    tickers = []
    for ticker in re.findall(r'\["([^"]+)",\s*"[^"]+",\s*"[^"]+",', html):
        if ticker.endswith(".PRE"):
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    return tickers


def series_values(frame, column):
    if frame is None or frame.empty or column not in frame:
        return []
    return [finite(value) for value in frame[column].tolist()]


def latest_value(values):
    for value in reversed(values):
        if value is not None:
            return value
    return None


def fetch_one(ticker):
    stock = yf.Ticker(ticker)
    fast_info = {}
    try:
        fast_info = dict(stock.fast_info or {})
    except Exception:
        fast_info = {}
    info = {}
    try:
        info = stock.get_info() or {}
    except Exception:
        info = {}

    price = finite(fast_info.get("last_price") or info.get("regularMarketPrice") or info.get("currentPrice"))
    currency = fast_info.get("currency") or info.get("currency") or "USD"
    market_cap = finite(fast_info.get("market_cap") or info.get("marketCap"))
    shares = finite(fast_info.get("shares") or info.get("sharesOutstanding"))
    volume_shares = finite(
        fast_info.get("last_volume")
        or fast_info.get("regular_market_volume")
        or info.get("regularMarketVolume")
        or info.get("volume")
    )

    intraday = None
    month = None
    try:
        intraday = stock.history(period="1d", interval="5m", auto_adjust=False, prepost=False)
    except Exception:
        intraday = None
    try:
        month = stock.history(period="1mo", interval="1d", auto_adjust=False, prepost=False)
    except Exception:
        month = None

    intraday_closes = series_values(intraday, "Close")
    month_closes = series_values(month, "Close")
    month_volumes = series_values(month, "Volume")

    close_now = latest_value(intraday_closes) or latest_value(month_closes) or price
    if close_now is not None:
        price = close_now
    if market_cap is None and shares is not None and price is not None:
        market_cap = shares * price

    change_1h = None
    if len(intraday_closes) >= 2 and close_now is not None:
        lookback = intraday_closes[-13] if len(intraday_closes) >= 13 else intraday_closes[0]
        change_1h = percent_move(close_now, lookback)

    change_percent = finite(fast_info.get("last_price_change_percent") or info.get("regularMarketChangePercent"))
    if change_percent is None and len(month_closes) >= 2:
        change_percent = percent_move(month_closes[-1], month_closes[-2])

    change_7d = None
    change_30d = None
    if month_closes:
        last = latest_value(month_closes)
        if last is not None:
            change_7d = percent_move(last, month_closes[max(0, len(month_closes) - 6)])
            change_30d = percent_move(last, month_closes[0])

    volume_value = None
    if price is not None:
        if volume_shares is not None:
            volume_value = price * volume_shares
        elif month_volumes:
            last_volume = latest_value(month_volumes)
            if last_volume is not None:
                volume_value = price * last_volume

    spark = normalize_spark(month_closes)

    return {
        "price": price,
        "currency": currency,
        "marketCap": market_cap,
        "volume": volume_value,
        "changePercent": change_percent,
        "change1h": change_1h,
        "change7d": change_7d,
        "change30d": change_30d,
        "sparkPoints": spark,
    }


def fetch_fx_rates():
    rates = {}
    for key, ticker in FX_TICKERS.items():
        try:
            fx = yf.Ticker(ticker)
            fast_info = {}
            try:
                fast_info = dict(fx.fast_info or {})
            except Exception:
                fast_info = {}
            info = {}
            try:
                info = fx.get_info() or {}
            except Exception:
                info = {}
            rate = finite(fast_info.get("last_price") or info.get("regularMarketPrice") or info.get("currentPrice"))
            if rate is not None and rate > 0:
                rates[key] = rate
        except Exception:
            continue
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Fetch only the first N listed tickers for smoke tests.")
    args = parser.parse_args()

    html = INDEX_HTML.read_text(encoding="utf-8")
    tickers = extract_listed_tickers(html)
    if args.limit:
        tickers = tickers[: args.limit]
    quotes = {}
    errors = {}
    fx_rates = fetch_fx_rates()

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] {ticker}")
        try:
            quotes[ticker] = fetch_one(ticker)
        except Exception as exc:
            errors[ticker] = str(exc)

    payload = {
        "meta": {
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "source": "Yahoo Finance via yfinance",
            "count": len(quotes),
            "fx": fx_rates,
            "errors": errors,
        },
        "quotes": quotes,
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    OUTFILE.write_text(text + "\n", encoding="utf-8")
    for mirror in MIRROR_DIRS:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / "quotes.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
