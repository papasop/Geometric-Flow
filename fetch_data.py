"""
fetch_data.py
=============
Fetch OHLC data for the AI Drone Portfolio components plus the
historical CNY/USD, HKD/USD, and EUR/USD spot rates, and write data.json next to
index.html.

Index inception: 2022-11-30 (ChatGPT public launch). Base = 100.

Index composition:
    Ondas                   ONDS       USD    10.00%
    Unusual Machines        UMAC       USD    10.00%
    Theon International     THEON.AS   EUR    10.00%
    Kopin                   KOPN       USD    10.00%
    Red Cat Holdings        RCAT       USD    10.00%
    Swarmer                 SWMR       USD    10.00%
    LightPath Technologies  LPTH       USD    10.00%
    Kratos Defense          KTOS       USD    10.00%
    Elbit Systems           ESLT       USD    10.00%
    AeroVironment           AVAV       USD    10.00%

Additional portfolio data pool:
    拼多多                  PDD        USD
    纽约时报                NYT        USD
    Rocket Lab              RKLB       USD
    比亚迪                  002594.SZ  CNY
    贵州茅台                600519.SS  CNY
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
    {"name": "LightPath Technologies", "short": "LPTH", "ticker": "LPTH", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Kratos Defense & Security Solutions", "short": "KTOS", "ticker": "KTOS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Elbit Systems", "short": "ESLT", "ticker": "ESLT", "ccy": "USD", "sleeve": "IL", "status": "active"},
    {"name": "AeroVironment", "short": "AVAV", "ticker": "AVAV", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Palantir Technologies", "short": "PLTR", "ticker": "PLTR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Rocket Lab", "short": "RKLB", "ticker": "RKLB", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "HawkEye 360", "short": "HAWK", "ticker": "HAWK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "BlackSky Technology", "short": "BKSY", "ticker": "BKSY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Spire Global", "short": "SPIR", "ticker": "SPIR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Planet Labs", "short": "PL", "ticker": "PL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "L3Harris Technologies", "short": "LHX", "ticker": "LHX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "速腾聚创", "short": "速腾聚创", "ticker": "02498.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "禾赛-W", "short": "禾赛", "ticker": "02525.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "图达通", "short": "图达通", "ticker": "02665.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "Ouster", "short": "OUST", "ticker": "OUST", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Salesforce", "short": "CRM", "ticker": "CRM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "MongoDB", "short": "MDB", "ticker": "MDB", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Pegasystems", "short": "PEGA", "ticker": "PEGA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Snowflake", "short": "SNOW", "ticker": "SNOW", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Elastic", "short": "ESTC", "ticker": "ESTC", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Confluent", "short": "CFLT", "ticker": "CFLT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Twilio", "short": "TWLO", "ticker": "TWLO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "CrowdStrike", "short": "CRWD", "ticker": "CRWD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "SentinelOne", "short": "S", "ticker": "S", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Palo Alto Networks", "short": "PANW", "ticker": "PANW", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Zscaler", "short": "ZS", "ticker": "ZS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Fortinet", "short": "FTNT", "ticker": "FTNT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Cloudflare", "short": "NET", "ticker": "NET", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Varonis Systems", "short": "VRNS", "ticker": "VRNS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Synopsys", "short": "SNPS", "ticker": "SNPS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Cadence Design Systems", "short": "CDNS", "ticker": "CDNS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "S&P Global", "short": "SPGI", "ticker": "SPGI", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Verisk Analytics", "short": "VRSK", "ticker": "VRSK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "CoStar Group", "short": "CSGP", "ticker": "CSGP", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Aurora Mobile", "short": "JG", "ticker": "JG", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "Agora", "short": "API", "ticker": "API", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "Braze", "short": "BRZE", "ticker": "BRZE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "AppLovin", "short": "APP", "ticker": "APP", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "HubSpot", "short": "HUBS", "ticker": "HUBS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Zoom", "short": "ZM", "ticker": "ZM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Bandwidth", "short": "BAND", "ticker": "BAND", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Five9", "short": "FIVN", "ticker": "FIVN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "RingCentral", "short": "RNG", "ticker": "RNG", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Atlassian", "short": "TEAM", "ticker": "TEAM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Adobe", "short": "ADBE", "ticker": "ADBE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Workday", "short": "WDAY", "ticker": "WDAY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "ADP", "short": "ADP", "ticker": "ADP", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Paycom", "short": "PAYC", "ticker": "PAYC", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Microsoft", "short": "MSFT", "ticker": "MSFT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Oracle", "short": "ORCL", "ticker": "ORCL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Alphabet", "short": "GOOG", "ticker": "GOOG", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Shopify", "short": "SHOP", "ticker": "SHOP", "ccy": "USD", "sleeve": "CA", "status": "active"},
    {"name": "SoFi Technologies", "short": "SOFI", "ticker": "SOFI", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Guidewire", "short": "GWRE", "ticker": "GWRE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Samsara", "short": "IOT", "ticker": "IOT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Rubrik", "short": "RBRK", "ticker": "RBRK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "ServiceNow", "short": "NOW", "ticker": "NOW", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Datadog", "short": "DDOG", "ticker": "DDOG", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Dynatrace", "short": "DT", "ticker": "DT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Constellation Software", "short": "CNSWF", "ticker": "CNSWF", "ccy": "USD", "sleeve": "CA", "status": "active"},
    {"name": "Topicus", "short": "TOITF", "ticker": "TOITF", "ccy": "USD", "sleeve": "CA", "status": "active"},
    {"name": "Roper Technologies", "short": "ROP", "ticker": "ROP", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Tyler Technologies", "short": "TYL", "ticker": "TYL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Bentley Systems", "short": "BSY", "ticker": "BSY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Veeva Systems", "short": "VEEV", "ticker": "VEEV", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Autodesk", "short": "ADSK", "ticker": "ADSK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Dassault Systemes", "short": "DASTY", "ticker": "DASTY", "ccy": "USD", "sleeve": "EU", "status": "active"},
    {"name": "SS&C Technologies", "short": "SSNC", "ticker": "SSNC", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Manhattan Associates", "short": "MANH", "ticker": "MANH", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Descartes Systems", "short": "DSGX", "ticker": "DSGX", "ccy": "USD", "sleeve": "CA", "status": "active"},
    {"name": "Workiva", "short": "WK", "ticker": "WK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Waystar", "short": "WAY", "ticker": "WAY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Toast", "short": "TOST", "ticker": "TOST", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "ServiceTitan", "short": "TTAN", "ticker": "TTAN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Procore Technologies", "short": "PCOR", "ticker": "PCOR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "NICE", "short": "NICE", "ticker": "NICE", "ccy": "USD", "sleeve": "IL", "status": "active"},
    {"name": "Decagon", "short": "DECAGON", "ticker": "DECAGON", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "\u7845\u57fa\u667a\u80fd", "short": "\u7845\u57fa\u667a\u80fd", "ticker": "SILICONINTELLIGENCE", "ccy": "USD", "sleeve": "CN", "status": "prelist"},
    {"name": "Moderna", "short": "MRNA", "ticker": "MRNA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u4e07\u6cf0\u751f\u7269", "short": "\u4e07\u6cf0\u751f\u7269", "ticker": "603392.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u8fbe\u5b89\u57fa\u56e0", "short": "\u8fbe\u5b89\u57fa\u56e0", "ticker": "002030.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "AbCellera Biologics", "short": "ABCL", "ticker": "ABCL", "ccy": "USD", "sleeve": "CA", "status": "active"},
    {"name": "Ginkgo Bioworks", "short": "DNA", "ticker": "DNA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Personalis", "short": "PSNL", "ticker": "PSNL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "CRISPR Therapeutics", "short": "CRSP", "ticker": "CRSP", "ccy": "USD", "sleeve": "CH", "status": "active"},
    {"name": "Recursion Pharmaceuticals", "short": "RXRX", "ticker": "RXRX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Absci", "short": "ABSI", "ticker": "ABSI", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Schrodinger", "short": "SDGR", "ticker": "SDGR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Incyte", "short": "INCY", "ticker": "INCY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "BeOne Medicines", "short": "ONC", "ticker": "ONC", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "Madrigal Pharmaceuticals", "short": "MDGL", "ticker": "MDGL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Insmed", "short": "INSM", "ticker": "INSM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "ACADIA Pharmaceuticals", "short": "ACAD", "ticker": "ACAD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Celcuity", "short": "CELC", "ticker": "CELC", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Revolution Medicines", "short": "RVMD", "ticker": "RVMD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Kymera Therapeutics", "short": "KYMR", "ticker": "KYMR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Summit Therapeutics", "short": "SMMT", "ticker": "SMMT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Rhythm Pharmaceuticals", "short": "RYTM", "ticker": "RYTM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Kodiak Sciences", "short": "KOD", "ticker": "KOD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Praxis Precision Medicines", "short": "PRAX", "ticker": "PRAX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Alkermes", "short": "ALKS", "ticker": "ALKS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "GRAIL", "short": "GRAL", "ticker": "GRAL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Edgewise Therapeutics", "short": "EWTX", "ticker": "EWTX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Arrowhead Pharmaceuticals", "short": "ARWR", "ticker": "ARWR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Ionis Pharmaceuticals", "short": "IONS", "ticker": "IONS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Immunovant", "short": "IMVT", "ticker": "IMVT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Immatics", "short": "IMTX", "ticker": "IMTX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "IDEAYA Biosciences", "short": "IDYA", "ticker": "IDYA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Roivant Sciences", "short": "ROIV", "ticker": "ROIV", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Abivax", "short": "ABVX", "ticker": "ABVX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Kiniksa Pharmaceuticals International", "short": "KNSA", "ticker": "KNSA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Replimune Group", "short": "REPL", "ticker": "REPL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Stoke Therapeutics", "short": "STOK", "ticker": "STOK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Structure Therapeutics", "short": "GPCR", "ticker": "GPCR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Nurix Therapeutics", "short": "NRIX", "ticker": "NRIX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Immunocore Holdings", "short": "IMCR", "ticker": "IMCR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Bicycle Therapeutics", "short": "BCYC", "ticker": "BCYC", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Crinetics Pharmaceuticals", "short": "CRNX", "ticker": "CRNX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Monte Rosa Therapeutics", "short": "GLUE", "ticker": "GLUE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Ultragenyx Pharmaceutical", "short": "RARE", "ticker": "RARE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Denali Therapeutics", "short": "DNLI", "ticker": "DNLI", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Viridian Therapeutics", "short": "VRDN", "ticker": "VRDN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Entrada Therapeutics", "short": "TRDA", "ticker": "TRDA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Rapport Therapeutics", "short": "RAPP", "ticker": "RAPP", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Terns Pharmaceuticals", "short": "TERN", "ticker": "TERN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "BioNTech", "short": "BNTX", "ticker": "BNTX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Spyre Therapeutics", "short": "SYRE", "ticker": "SYRE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Dianthus Therapeutics", "short": "DNTH", "ticker": "DNTH", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Prelude Therapeutics", "short": "PRLD", "ticker": "PRLD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "DBV Technologies", "short": "DBVT", "ticker": "DBVT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Disc Medicine", "short": "IRON", "ticker": "IRON", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Jade Biosciences", "short": "JBIO", "ticker": "JBIO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Taysha Gene Therapies", "short": "TSHA", "ticker": "TSHA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "BioCryst Pharmaceuticals", "short": "BCRX", "ticker": "BCRX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Ventyx Biosciences", "short": "VTYX", "ticker": "VTYX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "argenx", "short": "ARGX", "ticker": "ARGX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Design Therapeutics", "short": "DSGN", "ticker": "DSGN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Cerus", "short": "CERS", "ticker": "CERS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Neurogene", "short": "NGNE", "ticker": "NGNE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Tectonic Therapeutic", "short": "TECX", "ticker": "TECX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "CytomX Therapeutics", "short": "CTMX", "ticker": "CTMX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Celldex Therapeutics", "short": "CLDX", "ticker": "CLDX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Xencor", "short": "XNCR", "ticker": "XNCR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Centessa Pharmaceuticals", "short": "CNTA", "ticker": "CNTA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Xenon Pharmaceuticals", "short": "XENE", "ticker": "XENE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Sionna Therapeutics", "short": "SION", "ticker": "SION", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Contineum Therapeutics", "short": "CTNM", "ticker": "CTNM", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Sera Prognostics", "short": "SERA", "ticker": "SERA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Dyne Therapeutics", "short": "DYN", "ticker": "DYN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Alumis", "short": "ALMS", "ticker": "ALMS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Nuvalent", "short": "NUVL", "ticker": "NUVL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Pharvaris", "short": "PHVS", "ticker": "PHVS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Relay Therapeutics", "short": "RLAY", "ticker": "RLAY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Arcus Biosciences", "short": "RCUS", "ticker": "RCUS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "MapLight Therapeutics", "short": "MPLT", "ticker": "MPLT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "vTv Therapeutics", "short": "VTVT", "ticker": "VTVT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Wave Life Sciences", "short": "WVE", "ticker": "WVE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Sagimet Biosciences", "short": "SGMT", "ticker": "SGMT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Protara Therapeutics", "short": "TARA", "ticker": "TARA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "LB Pharmaceuticals", "short": "LBRX", "ticker": "LBRX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Enliven Therapeutics", "short": "ELVN", "ticker": "ELVN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "4D Molecular Therapeutics", "short": "FDMT", "ticker": "FDMT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Vir Biotechnology", "short": "VIR", "ticker": "VIR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "TScan Therapeutics", "short": "TCRX", "ticker": "TCRX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Sana Biotechnology", "short": "SANA", "ticker": "SANA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Skye Bioscience", "short": "SKYE", "ticker": "SKYE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "KALA BIO", "short": "KALA", "ticker": "KALA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Aligos Therapeutics", "short": "ALGS", "ticker": "ALGS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Tyra Biosciences", "short": "TYRA", "ticker": "TYRA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Acrivon Therapeutics", "short": "ACRV", "ticker": "ACRV", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Surrozen 股权认股权证", "short": "SRZNW", "ticker": "SRZNW", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Heron Therapeutics", "short": "HRTX", "ticker": "HRTX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "VYNE Therapeutics", "short": "VYNE", "ticker": "VYNE", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "MBX Biosciences", "short": "MBX", "ticker": "MBX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Arcellx", "short": "ACLX", "ticker": "ACLX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Vaxcyte", "short": "PCVX", "ticker": "PCVX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Viking Therapeutics", "short": "VKTX", "ticker": "VKTX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Opthea", "short": "OPT", "ticker": "OPT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Avidity Biosciences", "short": "RNA", "ticker": "RNA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Intellia Therapeutics", "short": "NTLA", "ticker": "NTLA", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Agios Pharmaceuticals", "short": "AGIO", "ticker": "AGIO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Cidara Therapeutics", "short": "CDTX", "ticker": "CDTX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Akero Therapeutics", "short": "AKRO", "ticker": "AKRO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Merus", "short": "MRUS", "ticker": "MRUS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Generation Bio", "short": "GBIO", "ticker": "GBIO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Skydio", "short": "SKYDIO", "ticker": "SKYDIO", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "Neros Technologies", "short": "NEROS", "ticker": "NEROS", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "SpaceX", "short": "SPACEX", "ticker": "SPACEX", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "Unseenlabs", "short": "UNSEENLABS", "ticker": "UNSEENLABS", "ccy": "EUR", "sleeve": "EU", "status": "prelist"},
    {"name": "\u62fc\u591a\u591a", "short": "PDD", "ticker": "PDD", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "\u7ebd\u7ea6\u65f6\u62a5", "short": "NYT", "ticker": "NYT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u6bd4\u4e9a\u8fea", "short": "BYD", "ticker": "002594.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u8d35\u5dde\u8305\u53f0", "short": "MOUTAI", "ticker": "600519.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u7f8e\u7684\u96c6\u56e2", "short": "\u7f8e\u7684", "ticker": "000333.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
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
    {"name": "iShares 20+ Year Treasury Bond ETF", "short": "TLT", "ticker": "TLT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Bitcoin", "short": "Bitcoin", "ticker": "BTC-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "Ethereum", "short": "ETH", "ticker": "ETH-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
    {"name": "SPDR Gold Shares", "short": "GLD", "ticker": "GLD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Apple", "short": "Apple", "ticker": "AAPL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Coca-Cola", "short": "\u53ef\u53e3\u53ef\u4e50", "ticker": "KO", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u6ce1\u6ce1\u739b\u7279", "short": "\u6ce1\u6ce1\u739b\u7279", "ticker": "9992.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u6c5f\u5357\u5e03\u8863", "short": "\u6c5f\u5357\u5e03\u8863", "ticker": "3306.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "MP Materials", "short": "MP", "ticker": "MP", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "USA Rare Earth", "short": "USAR", "ticker": "USAR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Critical Metals", "short": "CRML", "ticker": "CRML", "ccy": "USD", "sleeve": "US", "status": "active"},
]

TARGET_WEIGHTS = {
    "ONDS": 1 / 8,
    "UMAC": 1 / 8,
    "THEON.AS": 0.0,
    "KOPN": 1 / 8,
    "RCAT": 1 / 8,
    "SWMR": 1 / 8,
    "LPTH": 0.0,
    "KTOS": 1 / 8,
    "ESLT": 1 / 8,
    "AVAV": 1 / 8,
    "PLTR": 0.0,
    "RKLB": 0.0,
    "HAWK": 0.0,
    "BKSY": 0.0,
    "SPIR": 0.0,
    "PL": 0.0,
    "LHX": 0.0,
    "02498.HK": 0.0,
    "02525.HK": 0.0,
    "02665.HK": 0.0,
    "OUST": 0.0,
    "CRM": 0.0,
    "MDB": 0.0,
    "PEGA": 0.0,
    "SNOW": 0.0,
    "ESTC": 0.0,
    "CFLT": 0.0,
    "TWLO": 0.0,
    "CRWD": 0.0,
    "S": 0.0,
    "PANW": 0.0,
    "ZS": 0.0,
    "FTNT": 0.0,
    "NET": 0.0,
    "VRNS": 0.0,
    "SNPS": 0.0,
    "CDNS": 0.0,
    "SPGI": 0.0,
    "VRSK": 0.0,
    "CSGP": 0.0,
    "JG": 0.0,
    "API": 0.0,
    "BRZE": 0.0,
    "APP": 0.0,
    "HUBS": 0.0,
    "ZM": 0.0,
    "BAND": 0.0,
    "FIVN": 0.0,
    "RNG": 0.0,
    "TEAM": 0.0,
    "ADBE": 0.0,
    "WDAY": 0.0,
    "ADP": 0.0,
    "PAYC": 0.0,
    "MSFT": 0.0,
    "ORCL": 0.0,
    "GOOG": 0.0,
    "SHOP": 0.0,
    "SOFI": 0.0,
    "GWRE": 0.0,
    "IOT": 0.0,
    "RBRK": 0.0,
    "NOW": 0.0,
    "DDOG": 0.0,
    "DT": 0.0,
    "CNSWF": 0.0,
    "TOITF": 0.0,
    "ROP": 0.0,
    "TYL": 0.0,
    "BSY": 0.0,
    "VEEV": 0.0,
    "ADSK": 0.0,
    "DASTY": 0.0,
    "SSNC": 0.0,
    "MANH": 0.0,
    "DSGX": 0.0,
    "WK": 0.0,
    "WAY": 0.0,
    "TOST": 0.0,
    "TTAN": 0.0,
    "PCOR": 0.0,
    "NICE": 0.0,
    "DECAGON": 0.0,
    "SILICONINTELLIGENCE": 0.0,
    "MRNA": 0.0,
    "603392.SS": 0.0,
    "002030.SZ": 0.0,
    "ABCL": 0.0,
    "DNA": 0.0,
    "PSNL": 0.0,
    "CRSP": 0.0,
    "RXRX": 0.0,
    "ABSI": 0.0,
    "SDGR": 0.0,
    "INCY": 0.0,
    "ONC": 0.0,
    "MDGL": 0.0,
    "INSM": 0.0,
    "ACAD": 0.0,
    "CELC": 0.0,
    "RVMD": 0.0,
    "KYMR": 0.0,
    "SMMT": 0.0,
    "RYTM": 0.0,
    "KOD": 0.0,
    "PRAX": 0.0,
    "ALKS": 0.0,
    "GRAL": 0.0,
    "EWTX": 0.0,
    "ARWR": 0.0,
    "IONS": 0.0,
    "IMVT": 0.0,
    "IMTX": 0.0,
    "IDYA": 0.0,
    "ROIV": 0.0,
    "ABVX": 0.0,
    "KNSA": 0.0,
    "REPL": 0.0,
    "STOK": 0.0,
    "GPCR": 0.0,
    "NRIX": 0.0,
    "IMCR": 0.0,
    "BCYC": 0.0,
    "CRNX": 0.0,
    "GLUE": 0.0,
    "RARE": 0.0,
    "DNLI": 0.0,
    "VRDN": 0.0,
    "TRDA": 0.0,
    "RAPP": 0.0,
    "TERN": 0.0,
    "BNTX": 0.0,
    "SYRE": 0.0,
    "DNTH": 0.0,
    "PRLD": 0.0,
    "DBVT": 0.0,
    "IRON": 0.0,
    "JBIO": 0.0,
    "TSHA": 0.0,
    "BCRX": 0.0,
    "VTYX": 0.0,
    "ARGX": 0.0,
    "DSGN": 0.0,
    "CERS": 0.0,
    "NGNE": 0.0,
    "TECX": 0.0,
    "CTMX": 0.0,
    "CLDX": 0.0,
    "XNCR": 0.0,
    "CNTA": 0.0,
    "XENE": 0.0,
    "SION": 0.0,
    "CTNM": 0.0,
    "SERA": 0.0,
    "DYN": 0.0,
    "ALMS": 0.0,
    "NUVL": 0.0,
    "PHVS": 0.0,
    "RLAY": 0.0,
    "RCUS": 0.0,
    "MPLT": 0.0,
    "VTVT": 0.0,
    "WVE": 0.0,
    "SGMT": 0.0,
    "TARA": 0.0,
    "LBRX": 0.0,
    "ELVN": 0.0,
    "FDMT": 0.0,
    "VIR": 0.0,
    "TCRX": 0.0,
    "SANA": 0.0,
    "SKYE": 0.0,
    "KALA": 0.0,
    "ALGS": 0.0,
    "TYRA": 0.0,
    "ACRV": 0.0,
    "SRZNW": 0.0,
    "HRTX": 0.0,
    "VYNE": 0.0,
    "MBX": 0.0,
    "ACLX": 0.0,
    "PCVX": 0.0,
    "VKTX": 0.0,
    "OPT": 0.0,
    "RNA": 0.0,
    "NTLA": 0.0,
    "AGIO": 0.0,
    "CDTX": 0.0,
    "AKRO": 0.0,
    "MRUS": 0.0,
    "GBIO": 0.0,
    "SKYDIO": 0.0,
    "NEROS": 0.0,
    "SPACEX": 0.0,
    "UNSEENLABS": 0.0,
    "PDD": 0.0,
    "NYT": 0.0,
    "002594.SZ": 0.0,
    "600519.SS": 0.0,
    "000333.SZ": 0.0,
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
    "TLT": 0.0,
    "BTC-USD": 0.0,
    "ETH-USD": 0.0,
    "GLD": 0.0,
    "AAPL": 0.0,
    "KO": 0.0,
    "9992.HK": 0.0,
    "3306.HK": 0.0,
    "MP": 0.0,
    "USAR": 0.0,
    "CRML": 0.0,
}

SYNTHETIC_FALLBACKS = {
    "VTYX": {
        "start": "2022-11-30",
        "seed": 92321,
        "base": 9.0,
        "vol": 0.045,
        "drift": -0.00005,
        "note": "Yahoo Finance returned an empty series for VTYX; synthetic placeholder keeps the Baker Bros 13F holding visible.",
    },
    "OPT": {
        "start": "2022-11-30",
        "seed": 92322,
        "base": 4.0,
        "vol": 0.045,
        "drift": -0.00008,
        "note": "Yahoo Finance returned an empty series for OPT; synthetic placeholder keeps the Baker Bros historical holding visible.",
    },
    "CDTX": {
        "start": "2022-11-30",
        "seed": 92323,
        "base": 3.5,
        "vol": 0.045,
        "drift": -0.00006,
        "note": "Yahoo Finance returned an empty series for CDTX; synthetic placeholder keeps the Baker Bros historical holding visible.",
    },
    "AKRO": {
        "start": "2022-11-30",
        "seed": 92324,
        "base": 18.0,
        "vol": 0.045,
        "drift": 0.00005,
        "note": "Yahoo Finance returned an empty series for AKRO; synthetic placeholder keeps the Baker Bros historical holding visible.",
    },
    "MRUS": {
        "start": "2022-11-30",
        "seed": 92325,
        "base": 22.0,
        "vol": 0.045,
        "drift": 0.00002,
        "note": "Yahoo Finance returned an empty series for MRUS; synthetic placeholder keeps the Baker Bros historical holding visible.",
    },
    "GBIO": {
        "start": "2022-11-30",
        "seed": 92326,
        "base": 5.0,
        "vol": 0.045,
        "drift": -0.00010,
        "note": "Yahoo Finance returned an empty series for GBIO; synthetic placeholder keeps the Baker Bros historical holding visible.",
    },
}

LISTING_START_OVERRIDES = {
    # CRML was SZZL before Critical Metals completed its business combination.
    # Critical Metals began trading as CRML on Nasdaq on 2024-02-28, so trim
    # inherited SPAC history.
    "CRML": "2024-02-28",
    # USAR was IPXX before USA Rare Earth completed its SPAC business
    # combination. USA Rare Earth began trading as USAR on Nasdaq on
    # 2025-03-14, so trim inherited SPAC history.
    "USAR": "2025-03-14",
}

TICKER_CONTINUATIONS = {
    # Skydio has no public ticker yet. Try likely placeholders so the
    # pre-listing watch can activate automatically if either symbol appears.
    "SKYDIO": ["SKYDIO", "SKYD"],
    # Neros Technologies has no public ticker yet; these placeholders keep the
    # pre-listing watch ready to activate if either symbol becomes available.
    "NEROS": ["NEROS", "NROS"],
    # SpaceX is private; keep common placeholders ready for a future listing.
    "SPACEX": ["SPACEX", "SPACE"],
    # Unseenlabs is private; this placeholder keeps the watch ready to activate.
    "UNSEENLABS": ["UNSEENLABS"],
    # Hong Kong exchange codes are sometimes entered with a leading zero in the
    # UI, while Yahoo's chart endpoint uses the four-digit exchange code.
    "02498.HK": ["2498.HK"],
    "02525.HK": ["2525.HK"],
    "02665.HK": ["2665.HK"],
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
    out = pd.concat(parts).sort_index().groupby(level=0).last()
    listing_start = LISTING_START_OVERRIDES.get(ticker)
    if listing_start:
        out = out[out.index >= listing_start]
    return out


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
