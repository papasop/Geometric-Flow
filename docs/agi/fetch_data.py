"""
fetch_data.py
=============
Fetch OHLC data for the AI Drone Portfolio components plus the historical
CNY/USD, HKD/USD, and EUR/USD spot rates, and write data.json next to
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

import pandas as pd
import yfinance as yf


COMPONENTS = [
    # (CN name,    short,            ticker,         weight, ccy)
    ("Ondas",                  "ONDS",           "ONDS",      1 / 16, "USD"),
    ("Unusual Machines",       "UMAC",           "UMAC",      1 / 16, "USD"),
    ("Theon International",    "THEON",          "THEON.AS",  1 / 16, "EUR"),
    ("Kopin",                  "KOPN",           "KOPN",      1 / 16, "USD"),
    ("Red Cat Holdings",       "RCAT",           "RCAT",      1 / 16, "USD"),
    ("Swarmer",                "SWMR",           "SWMR",      1 / 16, "USD"),
    ("Draganfly",              "DPRO",           "DPRO",      1 / 16, "USD"),
    ("LightPath Technologies", "LPTH",           "LPTH",      1 / 16, "USD"),
    ("Safe Pro Group",         "SPAI",           "SPAI",      1 / 16, "USD"),
    ("Kratos Defense & Security Solutions", "KTOS", "KTOS", 1 / 16, "USD"),
    ("Palladyne AI",           "PDYN",           "PDYN",      1 / 16, "USD"),
    ("AeroVironment",          "AVAV",           "AVAV",      1 / 16, "USD"),
    ("Palantir Technologies",  "PLTR",           "PLTR",      1 / 16, "USD"),
    ("HawkEye 360",            "HAWK",           "HAWK",      1 / 16, "USD"),
    ("BlackSky Technology",    "BKSY",           "BKSY",      1 / 16, "USD"),
    ("Spire Global",           "SPIR",           "SPIR",      1 / 16, "USD"),
    ("Salesforce",             "CRM",            "CRM",       0.0, "USD"),
    ("CrowdStrike",            "CRWD",           "CRWD",      0.0, "USD"),
    ("Guidewire",              "GWRE",           "GWRE",      0.0, "USD"),
    ("Samsara",                "IOT",            "IOT",       0.0, "USD"),
    ("Rubrik",                 "RBRK",           "RBRK",      0.0, "USD"),
    ("ServiceNow",             "NOW",            "NOW",       0.0, "USD"),
    ("Datadog",                "DDOG",           "DDOG",      0.0, "USD"),
    ("Decagon",                "DECAGON",        "DECAGON",   0.0, "USD"),
    ("\u7845\u57fa\u667a\u80fd", "硅基智能",       "SILICONINTELLIGENCE", 0.0, "USD"),
    ("Moderna",                "MRNA",           "MRNA",      0.0, "USD"),
    ("\u4e07\u6cf0\u751f\u7269", "万泰生物",      "603392.SS", 0.0, "CNY"),
    ("\u8fbe\u5b89\u57fa\u56e0", "达安基因",      "002030.SZ", 0.0, "CNY"),
    ("\u62fc\u591a\u591a",      "PDD",            "PDD",       0.0, "USD"),
    ("\u7ebd\u7ea6\u65f6\u62a5", "NYT",           "NYT",       0.0, "USD"),
    ("\u6bd4\u4e9a\u8fea",      "BYD",            "002594.SZ", 0.0, "CNY"),
    ("\u8d35\u5dde\u8305\u53f0", "MOUTAI",        "600519.SS", 0.0, "CNY"),
    ("\u7f8e\u7684\u96c6\u56e2", "美的",          "000333.SZ", 0.0, "CNY"),
    ("TRON",                   "TRX",            "TRX-USD",   0.0, "USD"),
    ("Hyperliquid",            "HYPE",           "HYPE32196-USD", 0.0, "USD"),
    ("Sky Protocol",           "SKY",            "SKY33038-USD", 0.0, "USD"),
    ("\u5b81\u5fb7\u65f6\u4ee3", "CATL",          "300750.SZ", 0.0, "CNY"),
    ("\u6b23\u65fa\u8fbe",      "欣旺达",        "300207.SZ", 0.0, "CNY"),
    ("\u9f0e\u80dc\u65b0\u6750", "鼎胜新材",      "603876.SS", 0.0, "CNY"),
    ("\u6d77\u535a\u601d\u521b", "海博思创",      "688411.SS", 0.0, "CNY"),
    ("\u7ef4\u79d1\u6280\u672f", "维科技术",      "600152.SS", 0.0, "CNY"),
    ("\u5723\u9633\u80a1\u4efd", "圣阳股份",      "002580.SZ", 0.0, "CNY"),
    ("\u65b0\u9e3f\u57fa\u5730\u4ea7", "新地",    "0016.HK",  0.0, "HKD"),
    ("\u957f\u548c",            "长和",          "0001.HK",  0.0, "HKD"),
    ("\u5609\u91cc\u5efa\u8bbe", "嘉里建设",      "0683.HK",  0.0, "HKD"),
    ("\u6e2f\u706f-SS",         "港灯",          "2638.HK",  0.0, "HKD"),
    ("\u9999\u6e2f\u4e2d\u534e\u7164\u6c14", "中华煤气", "0003.HK", 0.0, "HKD"),
    ("\u4e2d\u7535\u63a7\u80a1", "中电",          "0002.HK",  0.0, "HKD"),
    ("\u4e2d\u94f6\u9999\u6e2f", "中银香港",      "2388.HK",  0.0, "HKD"),
    ("\u6c47\u4e30\u63a7\u80a1", "汇丰",          "0005.HK",  0.0, "HKD"),
    ("\u6e23\u6253\u96c6\u56e2", "渣打",          "2888.HK",  0.0, "HKD"),
    ("\u8fc8\u5bcc\u65f6",     "迈富时",        "2556.HK",  0.0, "HKD"),
    ("Bitcoin",                "Bitcoin",       "BTC-USD",  0.0, "USD"),
    ("Ethereum",               "ETH",           "ETH-USD",  0.0, "USD"),
    ("SPDR Gold Shares",       "GLD",           "GLD",      0.0, "USD"),
    ("Apple",                  "Apple",         "AAPL",     0.0, "USD"),
    ("Coca-Cola",              "可口可乐",      "KO",       0.0, "USD"),
    ("\u6ce1\u6ce1\u739b\u7279", "泡泡玛特",      "9992.HK",  0.0, "HKD"),
    ("\u745e\u58eb\u6cd5\u90ce", "CHF",           "CHF=X",    0.0, "USD"),
    ("\u4ee5\u8272\u5217\u8c22\u514b\u5c14", "ILS", "ILS=X", 0.0, "USD"),
]

TICKER_CONTINUATIONS = {
}


def fetch_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
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
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-11-30",
                    help="Index inception (default: ChatGPT public launch)")
    ap.add_argument("--end",   default=date.today().isoformat())
    ap.add_argument("--out",   default="data.json")
    args = ap.parse_args()

    # 1) Fetch all stock OHLC
    fetched = {}
    for name, short, ticker, weight, ccy in COMPONENTS:
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

    # 4) Align each component; back/forward-fill late-IPO gaps
    components_out = []
    for name, short, ticker, weight, ccy in COMPONENTS:
        df = fetched[ticker]
        inception = df.index.min() if len(df) else None
        # Reindex to common, then ffill (post-IPO holidays) + bfill (pre-IPO)
        df_aligned = df.reindex(common).ffill().bfill()
        if df_aligned.isna().any().any():
            raise SystemExit(f"Could not back-fill {ticker} \u2014 series is empty?")
        components_out.append({
            "name": name, "short": short, "ticker": ticker,
            "weight": weight, "ccy": ccy,
            "inception": inception,
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
        },
        "components": components_out,
        "fx": {
            "CNY": [{"date": idx, "rate": float(r)} for idx, r in fx_cny.items()],
            "HKD": [{"date": idx, "rate": float(r)} for idx, r in fx_hkd.items()],
            "EUR": [{"date": idx, "rate": float(r)} for idx, r in fx_eur.items()],
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


if __name__ == "__main__":
    main()
