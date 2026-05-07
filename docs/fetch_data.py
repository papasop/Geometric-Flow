"""
fetch_data.py
=============
Fetch OHLC data for the AI Drone Portfolio components plus the
historical CNY/USD, HKD/USD, and EUR/USD spot rates, and write data.json next to
index.html.

Index inception: 2022-11-30 (ChatGPT public launch). Base = 100.

Index composition:
    Ondas                   ONDS       USD    7.69%
    Unusual Machines        UMAC       USD    7.69%
    Theon International     THEON.AS   EUR    7.69%
    Kopin                   KOPN       USD    7.69%
    Red Cat Holdings        RCAT       USD    7.69%
    Swarmer                 SWMR       USD    7.69%
    Draganfly               DPRO       USD    7.69%
    LightPath Technologies  LPTH       USD    7.69%
    Safe Pro Group          SPAI       USD    7.69%
    Kratos Defense          KTOS       USD    7.69%
    Palladyne AI            PDYN       USD    7.69%
    AeroVironment           AVAV       USD    7.69%
    Palantir                PLTR       USD    7.69%

Additional portfolio data pool:
    拼多多                  PDD        USD
    纽约时报                NYT        USD
    比亚迪                  002594.SZ  CNY
    贵州茅台                600519.SS  CNY
    TRON                    TRX-USD    USD
    Hyperliquid             HYPE32196-USD USD
    Sky Protocol            SKY33038-USD USD

Calendar handling
-----------------
Each component trades on a different exchange with a different holiday
calendar. ONDS is used as the anchor calendar; all other components are
reindexed to that calendar.

For any component whose listing post-dates the start of the anchor calendar,
pre-IPO dates are back-filled with the first known close. This means that
component contributes a CONSTANT value to the index until it actually starts
trading. The distortion is bounded by its index weight and documented in the
JSON meta payload.

Usage:
    pip install yfinance pandas
    python fetch_data.py [--start 2022-11-30] [--end 2026-04-30]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


COMPONENTS = [
    {"name": "Ondas", "short": "ONDS", "ticker": "ONDS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Unusual Machines", "short": "UMAC", "ticker": "UMAC", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Theon International", "short": "THEON", "ticker": "THEON.AS", "ccy": "EUR", "sleeve": "EU", "status": "active"},
    {"name": "Kopin", "short": "KOPN", "ticker": "KOPN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Red Cat Holdings", "short": "RCAT", "ticker": "RCAT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Swarmer", "short": "SWMR", "ticker": "SWMR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Draganfly", "short": "DPRO", "ticker": "DPRO", "ccy": "USD", "sleeve": "CA", "status": "active"},
    {"name": "LightPath Technologies", "short": "LPTH", "ticker": "LPTH", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Safe Pro Group", "short": "SPAI", "ticker": "SPAI", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Kratos Defense & Security Solutions", "short": "KTOS", "ticker": "KTOS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Palladyne AI", "short": "PDYN", "ticker": "PDYN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "AeroVironment", "short": "AVAV", "ticker": "AVAV", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Palantir Technologies", "short": "PLTR", "ticker": "PLTR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Salesforce", "short": "CRM", "ticker": "CRM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "CrowdStrike", "short": "CRWD", "ticker": "CRWD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Guidewire", "short": "GWRE", "ticker": "GWRE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Samsara", "short": "IOT", "ticker": "IOT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Rubrik", "short": "RBRK", "ticker": "RBRK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "ServiceNow", "short": "NOW", "ticker": "NOW", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Decagon", "short": "DECAGON", "ticker": "DECAGON", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "\u7845\u57fa\u667a\u80fd", "short": "\u7845\u57fa\u667a\u80fd", "ticker": "SILICONINTELLIGENCE", "ccy": "USD", "sleeve": "CN", "status": "prelist"},
    {"name": "Skydio", "short": "SKYDIO", "ticker": "SKYDIO", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "Neros Technologies", "short": "NEROS", "ticker": "NEROS", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "\u62fc\u591a\u591a", "short": "PDD", "ticker": "PDD", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "\u7ebd\u7ea6\u65f6\u62a5", "short": "NYT", "ticker": "NYT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u6bd4\u4e9a\u8fea", "short": "BYD", "ticker": "002594.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u8d35\u5dde\u8305\u53f0", "short": "MOUTAI", "ticker": "600519.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u7f8e\u7684\u96c6\u56e2", "short": "\u7f8e\u7684", "ticker": "000333.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "TRON", "short": "TRX", "ticker": "TRX-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "Hyperliquid", "short": "HYPE", "ticker": "HYPE32196-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "Sky Protocol", "short": "SKY", "ticker": "SKY33038-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "\u5b81\u5fb7\u65f6\u4ee3", "short": "CATL", "ticker": "300750.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u6b23\u65fa\u8fbe", "short": "\u6b23\u65fa\u8fbe", "ticker": "300207.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u9f0e\u80dc\u65b0\u6750", "short": "\u9f0e\u80dc\u65b0\u6750", "ticker": "603876.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u6d77\u535a\u601d\u521b", "short": "\u6d77\u535a\u601d\u521b", "ticker": "688411.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u7ef4\u79d1\u6280\u672f", "short": "\u7ef4\u79d1\u6280\u672f", "ticker": "600152.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u5723\u9633\u80a1\u4efd", "short": "\u5723\u9633\u80a1\u4efd", "ticker": "002580.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u65b0\u9e3f\u57fa\u5730\u4ea7", "short": "\u65b0\u5730", "ticker": "0016.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u957f\u548c", "short": "\u957f\u548c", "ticker": "0001.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u5609\u91cc\u5efa\u8bbe", "short": "\u5609\u91cc\u5efa\u8bbe", "ticker": "0683.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6e2f\u706f-SS", "short": "\u6e2f\u706f", "ticker": "2638.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u9999\u6e2f\u4e2d\u534e\u7164\u6c14", "short": "\u4e2d\u534e\u7164\u6c14", "ticker": "0003.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u4e2d\u7535\u63a7\u80a1", "short": "\u4e2d\u7535", "ticker": "0002.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u4e2d\u94f6\u9999\u6e2f", "short": "\u4e2d\u94f6\u9999\u6e2f", "ticker": "2388.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6c47\u4e30\u63a7\u80a1", "short": "\u6c47\u4e30", "ticker": "0005.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6e23\u6253\u96c6\u56e2", "short": "\u6e23\u6253", "ticker": "2888.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u8fc8\u5bcc\u65f6", "short": "\u8fc8\u5bcc\u65f6", "ticker": "2556.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "Bitcoin", "short": "Bitcoin", "ticker": "BTC-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "Ethereum", "short": "ETH", "ticker": "ETH-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "SPDR Gold Shares", "short": "GLD", "ticker": "GLD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Apple", "short": "Apple", "ticker": "AAPL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Coca-Cola", "short": "\u53ef\u53e3\u53ef\u4e50", "ticker": "KO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u6ce1\u6ce1\u739b\u7279", "short": "\u6ce1\u6ce1\u739b\u7279", "ticker": "9992.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u745e\u58eb\u6cd5\u90ce", "short": "CHF", "ticker": "CHF=X", "ccy": "USD", "sleeve": "FX", "status": "active"},
    {"name": "\u4ee5\u8272\u5217\u8c22\u514b\u5c14", "short": "ILS", "ticker": "ILS=X", "ccy": "USD", "sleeve": "FX", "status": "active"},
]

TARGET_WEIGHTS = {
    "ONDS": 1 / 13,
    "UMAC": 1 / 13,
    "THEON.AS": 1 / 13,
    "KOPN": 1 / 13,
    "RCAT": 1 / 13,
    "SWMR": 1 / 13,
    "DPRO": 1 / 13,
    "LPTH": 1 / 13,
    "SPAI": 1 / 13,
    "KTOS": 1 / 13,
    "PDYN": 1 / 13,
    "AVAV": 1 / 13,
    "PLTR": 1 / 13,
    "CRM": 0.0,
    "CRWD": 0.0,
    "GWRE": 0.0,
    "IOT": 0.0,
    "RBRK": 0.0,
    "NOW": 0.0,
    "DECAGON": 0.0,
    "SILICONINTELLIGENCE": 0.0,
    "SKYDIO": 0.0,
    "NEROS": 0.0,
    "PDD": 0.0,
    "NYT": 0.0,
    "002594.SZ": 0.0,
    "600519.SS": 0.0,
    "000333.SZ": 0.0,
    "TRX-USD": 0.0,
    "HYPE32196-USD": 0.0,
    "SKY33038-USD": 0.0,
    "300750.SZ": 0.0,
    "300207.SZ": 0.0,
    "603876.SS": 0.0,
    "688411.SS": 0.0,
    "600152.SS": 0.0,
    "002580.SZ": 0.0,
    "0016.HK": 0.0,
    "0001.HK": 0.0,
    "0683.HK": 0.0,
    "2638.HK": 0.0,
    "0003.HK": 0.0,
    "0002.HK": 0.0,
    "2388.HK": 0.0,
    "0005.HK": 0.0,
    "2888.HK": 0.0,
    "2556.HK": 0.0,
    "BTC-USD": 0.0,
    "ETH-USD": 0.0,
    "GLD": 0.0,
    "AAPL": 0.0,
    "KO": 0.0,
    "9992.HK": 0.0,
    "CHF=X": 0.0,
    "ILS=X": 0.0,
}

SYNTHETIC_FALLBACKS = {}

TICKER_CONTINUATIONS = {
    # Skydio has no public ticker yet. Try likely placeholders so the
    # pre-listing watch can activate automatically if either symbol appears.
    "SKYDIO": ["SKYDIO", "SKYD"],
    # Neros Technologies has no public ticker yet; these placeholders keep the
    # pre-listing watch ready to activate if either symbol becomes available.
    "NEROS": ["NEROS", "NROS"],
}


def mulberry32(seed: int):
    def rand() -> float:
        nonlocal seed
        seed = (seed + 0x6D2B79F5) & 0xFFFFFFFF
        t = seed
        t = (t ^ (t >> 15)) * (t | 1)
        t &= 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rand


def generate_synthetic_ohlc(index: pd.Index, seed: int, base: float,
                            daily_vol: float, drift: float) -> pd.DataFrame:
    rng = mulberry32(seed)
    close = base
    rows = []
    for idx in index:
        ret = (rng() - 0.5) * 2 * daily_vol + drift
        open_ = close * (1 + (rng() - 0.5) * 0.004)
        new_close = open_ * (1 + ret)
        wick_frac = (rng() * 0.5 + 0.2) * daily_vol
        high = max(open_, new_close) * (1 + wick_frac)
        low = min(open_, new_close) * (1 - wick_frac)
        rows.append({"Open": open_, "High": high, "Low": low, "Close": new_close})
        close = new_close
    return pd.DataFrame(rows, index=index)


def generate_synthetic_yield(index: pd.Index, seed: int, base: float,
                             daily_vol: float, drift: float,
                             floor: float, ceiling: float) -> pd.DataFrame:
    rng = mulberry32(seed)
    close = base
    rows = []
    for _ in index:
        change = (rng() - 0.5) * 2 * daily_vol + drift
        open_ = close + (rng() - 0.5) * daily_vol * 0.35
        close = min(ceiling, max(floor, open_ + change))
        wick = (rng() * 0.5 + 0.2) * daily_vol
        high = min(ceiling, max(open_, close) + wick)
        low = max(floor, min(open_, close) - wick)
        rows.append({"Open": open_, "High": high, "Low": low, "Close": close})
    return pd.DataFrame(rows, index=index)


def fetch_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, threads=False)
    if df.empty:
        print(f"  WARNING: empty result for {ticker}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = df.index.strftime("%Y-%m-%d")
    return df


def fetch_component_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    tickers = TICKER_CONTINUATIONS.get(ticker, [ticker])
    parts = []
    for source_ticker in tickers:
        df = fetch_ohlc(source_ticker, start, end)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    return pd.concat(parts).sort_index().groupby(level=0).last()


def fetch_fx(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, threads=False)
    if df.empty:
        raise SystemExit(f"FX series unavailable: {ticker}")
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = s.index.strftime("%Y-%m-%d")
    return s


def to_records(df: pd.DataFrame) -> list:
    return [
        {"date": idx,
         "open":  float(r["Open"]),
         "high":  float(r["High"]),
         "low":   float(r["Low"]),
         "close": float(r["Close"])}
        for idx, r in df.iterrows()
    ]


def normalize_tnx(df: pd.DataFrame) -> pd.DataFrame:
    # Yahoo's ^TNX is often quoted as yield * 10 (e.g. 43.5 = 4.35%).
    if df.empty:
        return df
    out = df.copy()
    if out["Close"].median() > 20:
        out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]] / 10.0
    return out


def main():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-11-30",
                    help="Index inception (default: ChatGPT public launch)")
    ap.add_argument("--end",   default=date.today().isoformat())
    ap.add_argument("--out",   default=str(script_dir / "data.json"))
    args = ap.parse_args()

    # 1) Fetch all stock OHLC
    fetched = {}
    for component in COMPONENTS:
        name, short, ticker, ccy = component["name"], component["short"], component["ticker"], component["ccy"]
        print(f"Fetching {short:15s} ({ticker:12s}, {ccy}) ...")
        fetched[ticker] = fetch_component_ohlc(ticker, args.start, args.end)

    # 2) Anchor calendar = ONDS
    anchor = fetched["ONDS"]
    if anchor.empty:
        raise SystemExit("ONDS data missing; cannot establish anchor calendar.")
    common = anchor.index
    print(f"Anchor calendar: {len(common)} ONDS trading days "
          f"({common[0]} \u2192 {common[-1]})")

    # 3) Fetch FX, align to anchor calendar
    print("Fetching FX  CNY=X (CNY per USD) ...")
    fx_cny = fetch_fx("CNY=X", args.start, args.end)
    print("Fetching FX  HKD=X (HKD per USD) ...")
    fx_hkd = fetch_fx("HKD=X", args.start, args.end)
    print("Fetching FX  USDEUR=X (EUR per USD) ...")
    fx_eur = fetch_fx("USDEUR=X", args.start, args.end)
    fx_cny = fx_cny.reindex(common).ffill().bfill()
    fx_hkd = fx_hkd.reindex(common).ffill().bfill()
    fx_eur = fx_eur.reindex(common).ffill().bfill()

    # 3b) Fetch benchmark, aligned to the same anchor calendar.
    print("Fetching benchmark NASDAQ (^IXIC) ...")
    nasdaq = fetch_ohlc("^IXIC", args.start, args.end)
    if nasdaq.empty:
        print("  WARNING: empty benchmark result for ^IXIC")
    nasdaq_aligned = nasdaq.reindex(common).ffill().bfill()
    if nasdaq_aligned.isna().any().any():
        raise SystemExit("NASDAQ benchmark data missing; cannot align ^IXIC.")
    print("Fetching benchmark Bitcoin (BTC-USD) ...")
    bitcoin = fetch_ohlc("BTC-USD", args.start, args.end)
    if bitcoin.empty:
        print("  WARNING: empty benchmark result for BTC-USD")
    bitcoin_aligned = bitcoin.reindex(common).ffill().bfill()
    if bitcoin_aligned.isna().any().any():
        raise SystemExit("Bitcoin benchmark data missing; cannot align BTC-USD.")

    # 3c) Macro yield pane. US10Y uses Yahoo ^TNX when available; CN10Y
    # is a local synthetic curve because yfinance does not expose a stable
    # China 10Y government-bond ticker.
    print("Fetching yield US10Y (^TNX) ...")
    us10y = normalize_tnx(fetch_ohlc("^TNX", args.start, args.end))
    if us10y.empty:
        print("  WARNING: empty yield result for ^TNX; using synthetic fallback")
        us10y = generate_synthetic_yield(common, 2026, 3.70, 0.035, 0.0002, 2.0, 6.0)
    us10y_aligned = us10y.reindex(common).ffill().bfill()
    if us10y_aligned.isna().any().any():
        raise SystemExit("US10Y yield data missing; cannot align ^TNX.")
    print("Building yield CN10Y synthetic curve ...")
    cn10y_aligned = generate_synthetic_yield(common, 1010, 2.85, 0.025, -0.0005, 1.5, 4.0)
    # 4) Align each component; back/forward-fill late-IPO gaps
    components_out = []
    pending_components = []
    fallback_notes = {}
    for component in COMPONENTS:
        name, short, ticker, ccy = component["name"], component["short"], component["ticker"], component["ccy"]
        status = component.get("status", "active")
        df = fetched[ticker]
        if status == "prelist" and df.empty:
            pending_components.append({
                "name": name,
                "short": short,
                "ticker": ticker,
                "ccy": ccy,
                "sleeve": component.get("sleeve"),
                "status": "prelist",
                "note": "Pre-listing watch code; excluded until exchange data exists.",
            })
            print(f"  {short:15s}: pre-listing watch only; excluded until data exists")
            continue
        if status == "prelist" and not df.empty:
            status = "active"
        if df.empty and ticker in SYNTHETIC_FALLBACKS:
            fb = SYNTHETIC_FALLBACKS[ticker]
            fb_index = common[common >= fb["start"]]
            if len(fb_index) == 0:
                fb_index = common[-1:]
            df = generate_synthetic_ohlc(
                fb_index, fb["seed"], fb["base"], fb["vol"], fb["drift"]
            )
            fallback_notes[ticker] = fb["note"]
            print(f"  {short:15s}: using synthetic fallback from {fb_index[0]}")
        inception = df.index.min() if len(df) else None
        # Reindex to common, then ffill (post-IPO holidays) + bfill (pre-IPO)
        df_aligned = df.reindex(common).ffill().bfill()
        if df_aligned.isna().any().any():
            raise SystemExit(f"Could not back-fill {ticker} \u2014 series is empty?")
        components_out.append({
            "name": name, "short": short, "ticker": ticker,
            "weight": TARGET_WEIGHTS.get(ticker, 0.0), "ccy": ccy,
            "sleeve": component.get("sleeve"),
            "status": status,
            "inception": inception,
            "synthetic_fallback": fallback_notes.get(ticker),
            "data": to_records(df_aligned),
        })
        late = " (back-filled pre-IPO!)" if inception and inception > common[0] else ""
        print(f"  {short:15s}: aligned to {len(df_aligned)} days, "
              f"real history from {inception}{late}")

    # 5) Write JSON
    payload = {
        "meta": {
            "start": args.start,
            "end":   args.end,
            "n":     len(common),
            "calendar_anchors": ["ONDS"],
            "fx_unit_CNY": "CNY per USD (yfinance: CNY=X)",
            "fx_unit_HKD": "HKD per USD (yfinance: HKD=X)",
            "fx_unit_EUR": "EUR per USD (yfinance: USDEUR=X)",
            "note": ("Pre-inception dates of late-listed components are "
                     "back-filled with their first known close. The index "
                     "therefore inherits a constant contribution from those "
                     "components prior to their actual IPO."),
            "synthetic_fallbacks": fallback_notes,
            "pending_components": pending_components,
            "weighting_policy": "Data pool for portfolio switching. Active UI portfolios override ticker weights.",
        },
        "components": components_out,
        "fx": {
            "CNY": [{"date": idx, "rate": float(r)} for idx, r in fx_cny.items()],
            "HKD": [{"date": idx, "rate": float(r)} for idx, r in fx_hkd.items()],
            "EUR": [{"date": idx, "rate": float(r)} for idx, r in fx_eur.items()],
        },
        "benchmarks": {
            "NASDAQ": {
                "name": "Nasdaq Composite",
                "ticker": "^IXIC",
                "data": to_records(nasdaq_aligned),
            },
            "BITCOIN": {
                "name": "Bitcoin",
                "ticker": "BTC-USD",
                "data": to_records(bitcoin_aligned),
            },
        },
        "yields": {
            "CN10Y": {
                "name": "China 10Y Government Bond Yield",
                "ticker": "CN10Y_SYNTH",
                "synthetic_fallback": "Synthetic local curve; yfinance has no stable China 10Y ticker.",
                "data": to_records(cn10y_aligned),
            },
            "US10Y": {
                "name": "US 10Y Treasury Yield",
                "ticker": "^TNX",
                "data": to_records(us10y_aligned),
            },
        },
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    weights_sum = sum(c["weight"] for c in components_out)
    print(f"\nWrote {args.out}")
    print(f"  Components: {len(components_out)} | weights sum: {weights_sum:.3f}")
    print(f"  CNY/USD:    {fx_cny.iloc[0]:.4f} \u2192 {fx_cny.iloc[-1]:.4f}  "
          f"({(fx_cny.iloc[-1]/fx_cny.iloc[0] - 1) * 100:+.2f}%)")
    print(f"  HKD/USD:    {fx_hkd.iloc[0]:.4f} \u2192 {fx_hkd.iloc[-1]:.4f}  "
          f"({(fx_hkd.iloc[-1]/fx_hkd.iloc[0] - 1) * 100:+.2f}%)")
    print(f"  EUR/USD:    {fx_eur.iloc[0]:.4f} \u2192 {fx_eur.iloc[-1]:.4f}  "
          f"({(fx_eur.iloc[-1]/fx_eur.iloc[0] - 1) * 100:+.2f}%)")
    print(f"  NASDAQ:     {nasdaq_aligned['Close'].iloc[0]:.2f} \u2192 "
          f"{nasdaq_aligned['Close'].iloc[-1]:.2f}  "
          f"({(nasdaq_aligned['Close'].iloc[-1]/nasdaq_aligned['Close'].iloc[0] - 1) * 100:+.2f}%)")
    print(f"  Bitcoin:    {bitcoin_aligned['Close'].iloc[0]:.2f} \u2192 "
          f"{bitcoin_aligned['Close'].iloc[-1]:.2f}  "
          f"({(bitcoin_aligned['Close'].iloc[-1]/bitcoin_aligned['Close'].iloc[0] - 1) * 100:+.2f}%)")
    print(f"  CN10Y:      {cn10y_aligned['Close'].iloc[0]:.3f}% \u2192 "
          f"{cn10y_aligned['Close'].iloc[-1]:.3f}%")
    print(f"  US10Y:      {us10y_aligned['Close'].iloc[0]:.3f}% \u2192 "
          f"{us10y_aligned['Close'].iloc[-1]:.3f}%")


if __name__ == "__main__":
    main()
