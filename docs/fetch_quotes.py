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
    "USDHKD": "HKD=X",
    "USDKRW": "KRW=X",
    "USDJPY": "JPY=X",
    "USDTWD": "TWD=X",
    "USDGBP": "GBP=X",
    "USDCAD": "CAD=X",
    "USDSEK": "SEK=X",
}
FALLBACK_SHARES_OUTSTANDING = {
    # SJ Semiconductor (688820.SH) is newly listed; Yahoo can return a price
    # without market cap or shares. SSE listing notice: 1,862,774,097 shares.
    "688820.SS": 1_862_774_097,
}


def inferred_currency(ticker):
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return "TWD"
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return "KRW"
    if ticker.endswith(".T") or ticker.endswith(".JP"):
        return "JPY"
    if ticker.endswith(".HK"):
        return "HKD"
    if ticker.endswith(".SS") or ticker.endswith(".SZ"):
        return "CNY"
    if ticker.endswith(".L"):
        return "GBp"
    if ticker.endswith(".TO") or ticker.endswith(".V"):
        return "CAD"
    if ticker.endswith(".ST"):
        return "SEK"
    return "USD"


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


def extract_js_array_block(html, const_name):
    match = re.search(rf"const\s+{re.escape(const_name)}\s*=\s*\[", html)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(html)):
        char = html[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"', "`"):
            quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return html[start : index + 1]
    return ""


def looks_like_ticker(value):
    if value.endswith(".PRE"):
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{1,8}(?:\.[A-Z]{1,4})?", value))


def extract_listed_tickers(html):
    tickers = []
    blocks = [
        extract_js_array_block(html, "BASE_COMPANIES"),
        extract_js_array_block(html, "AIBOTT_WATCHLIST"),
    ]
    if not any(blocks):
        blocks = [html]
    for block in blocks:
        for ticker in re.findall(r'\[\s*"([^"]+)"\s*,', block):
            if not looks_like_ticker(ticker):
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


def clean_values(values):
    return [value for value in (finite(value) for value in values) if value is not None]


def close_points(frame):
    if frame is None or frame.empty or "Close" not in frame:
        return None
    points = []
    for index, value in zip(frame.index, frame["Close"].tolist()):
        close = finite(value)
        if close is None:
            continue
        try:
            date = index.to_pydatetime().date().isoformat()
        except AttributeError:
            date = str(index)[:10]
        points.append({"date": date, "close": close})
    return points or None


def previous_value(values):
    clean = clean_values(values)
    if len(clean) < 2:
        return None
    return clean[-2]


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
    currency = fast_info.get("currency") or info.get("currency") or inferred_currency(ticker)
    market_cap = finite(fast_info.get("market_cap") or info.get("marketCap"))
    market_cap_currency = currency
    shares = finite(fast_info.get("shares") or info.get("sharesOutstanding"))
    if shares is None:
        shares = FALLBACK_SHARES_OUTSTANDING.get(ticker)
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
    try:
        if ticker in {"NBIS", "NVDA"}:
            history = stock.history(period="max", interval="1d", auto_adjust=False, prepost=False)
        else:
            history = stock.history(period="5y", interval="1wk", auto_adjust=False, prepost=False)
    except Exception:
        history = None

    intraday_closes = series_values(intraday, "Close")
    month_closes = series_values(month, "Close")
    month_volumes = series_values(month, "Volume")

    close_now = latest_value(intraday_closes) or latest_value(month_closes) or price
    if close_now is not None:
        price = close_now
    if shares is not None and price is not None:
        market_cap = shares * price
        if currency == "GBp":
            market_cap = market_cap / 100
            market_cap_currency = "GBP"
    elif currency == "GBp" and market_cap is not None:
        market_cap_currency = "GBP"

    change_1h = None
    if len(intraday_closes) >= 2 and close_now is not None:
        lookback = intraday_closes[-13] if len(intraday_closes) >= 13 else intraday_closes[0]
        change_1h = percent_move(close_now, lookback)

    change_percent = None
    previous_close = previous_value(month_closes)
    if previous_close is not None and close_now is not None:
        change_percent = percent_move(close_now, previous_close)
    if change_percent is None:
        change_percent = finite(info.get("regularMarketChangePercent") or fast_info.get("last_price_change_percent"))

    change_7d = None
    change_30d = None
    clean_month_closes = clean_values(month_closes)
    if clean_month_closes:
        last = close_now or clean_month_closes[-1]
        if last is not None:
            change_7d = percent_move(last, clean_month_closes[max(0, len(clean_month_closes) - 6)])
            change_30d = percent_move(last, clean_month_closes[0])

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
        "marketCapCurrency": market_cap_currency,
        "volume": volume_value,
        "changePercent": change_percent,
        "change1h": change_1h,
        "change7d": change_7d,
        "change30d": change_30d,
        "sparkPoints": spark,
        "intradayClosePoints": close_points(intraday),
        "closePoints": close_points(month),
        "historyClosePoints": close_points(history),
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
    parser.add_argument("--tickers", nargs="*", default=None, help="Fetch only these ticker symbols.")
    parser.add_argument("--missing-only", action="store_true", help="Fetch only listed tickers missing from the existing output.")
    parser.add_argument("--merge", action="store_true", help="Merge fetched quotes into the existing output instead of replacing it.")
    args = parser.parse_args()

    html = INDEX_HTML.read_text(encoding="utf-8")
    tickers = extract_listed_tickers(html)
    existing_payload = {}
    if OUTFILE.exists():
        try:
            existing_payload = json.loads(OUTFILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_payload = {}
    existing_quotes = existing_payload.get("quotes") if isinstance(existing_payload.get("quotes"), dict) else {}
    existing_meta = existing_payload.get("meta") if isinstance(existing_payload.get("meta"), dict) else {}
    if args.tickers:
        requested = [ticker.strip().upper() for ticker in args.tickers if ticker.strip()]
        tickers = [ticker for ticker in requested if looks_like_ticker(ticker)]
    if args.missing_only:
        tickers = [ticker for ticker in tickers if ticker not in existing_quotes]
    if args.limit:
        tickers = tickers[: args.limit]
    quotes = dict(existing_quotes) if args.merge or args.missing_only or args.tickers else {}
    errors = dict(existing_meta.get("errors") or {}) if args.merge or args.missing_only or args.tickers else {}
    fx_rates = fetch_fx_rates()
    if (args.merge or args.missing_only or args.tickers) and not fx_rates:
        fx_rates = existing_meta.get("fx") or {}

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] {ticker}")
        try:
            quotes[ticker] = fetch_one(ticker)
            errors.pop(ticker, None)
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
