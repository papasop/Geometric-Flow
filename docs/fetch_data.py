"""
fetch_data.py
=============
Fetch OHLC data for the AI Drone Portfolio components plus the
historical CNY/USD, HKD/USD, EUR/USD, and JPY/USD spot rates, and write data.json next to
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
from datetime import date, timedelta
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
    {"name": "Terra Drone", "short": "278A", "ticker": "278A.T", "ccy": "JPY", "sleeve": "JP", "status": "active"},
    {"name": "Palantir Technologies", "short": "PLTR", "ticker": "PLTR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Rocket Lab", "short": "RKLB", "ticker": "RKLB", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "HawkEye 360", "short": "HAWK", "ticker": "HAWK", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "BlackSky Technology", "short": "BKSY", "ticker": "BKSY", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Spire Global", "short": "SPIR", "ticker": "SPIR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Planet Labs", "short": "PL", "ticker": "PL", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "L3Harris Technologies", "short": "LHX", "ticker": "LHX", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "速腾聚创", "short": "速腾聚创", "ticker": "02498.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
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
    {"name": "Nebius Group", "short": "NBIS", "ticker": "NBIS", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "CoreWeave", "short": "CRWV", "ticker": "CRWV", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "IREN", "short": "IREN", "ticker": "IREN", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Applied Digital", "short": "APLD", "ticker": "APLD", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "TeraWulf", "short": "WULF", "ticker": "WULF", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Hut 8", "short": "HUT", "ticker": "HUT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Core Scientific", "short": "CORZ", "ticker": "CORZ", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Digital Realty", "short": "DLR", "ticker": "DLR", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u4e16\u7eaa\u4e92\u8fde", "short": "VNET", "ticker": "VNET", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "Lambda Labs", "short": "LAMBDA", "ticker": "LAMBDA", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "ClickHouse", "short": "CLICKHOUSE", "ticker": "CLICKHOUSE", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "Cursor", "short": "CURSOR", "ticker": "CURSOR", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "Lotus AI", "short": "LOTUSAI", "ticker": "LOTUSAI", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "Slingshot AI", "short": "SLINGSHOTAI", "ticker": "SLINGSHOTAI", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "GetDynasty", "short": "GETDYNASTY", "ticker": "GETDYNASTY", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "E2B", "short": "E2B", "ticker": "E2B", "ccy": "USD", "sleeve": "US", "status": "prelist"},
    {"name": "\u62fc\u591a\u591a", "short": "PDD", "ticker": "PDD", "ccy": "USD", "sleeve": "CN", "status": "active"},
    {"name": "\u7ebd\u7ea6\u65f6\u62a5", "short": "NYT", "ticker": "NYT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "\u8d35\u5dde\u8305\u53f0", "short": "MOUTAI", "ticker": "600519.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u7f8e\u7684\u96c6\u56e2", "short": "\u7f8e\u7684", "ticker": "000333.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u65b0\u9e3f\u57fa\u5730\u4ea7", "short": "\u65b0\u5730", "ticker": "0016.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u957f\u548c", "short": "\u957f\u548c", "ticker": "0001.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u5609\u91cc\u5efa\u8bbe", "short": "\u5609\u91cc\u5efa\u8bbe", "ticker": "0683.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6e2f\u706f-SS", "short": "\u6e2f\u706f", "ticker": "2638.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u9999\u6e2f\u4e2d\u534e\u7164\u6c14", "short": "\u4e2d\u534e\u7164\u6c14", "ticker": "0003.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u4e2d\u7535\u63a7\u80a1", "short": "\u4e2d\u7535", "ticker": "0002.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u4e2d\u94f6\u9999\u6e2f", "short": "\u4e2d\u94f6\u9999\u6e2f", "ticker": "2388.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6c47\u4e30\u63a7\u80a1", "short": "\u6c47\u4e30", "ticker": "0005.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6e23\u6253\u96c6\u56e2", "short": "\u6e23\u6253", "ticker": "2888.HK", "ccy": "HKD", "sleeve": "HK", "status": "active"},
    {"name": "\u6d25\u4e0a\u673a\u5e8a\u4e2d\u56fd", "short": "\u6d25\u4e0a\u673a\u5e8a", "ticker": "1651.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u91d1\u529b\u6c38\u78c1", "short": "\u91d1\u529b\u6c38\u78c1", "ticker": "300748.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u5929\u5cb3\u5148\u8fdb", "short": "\u5929\u5cb3\u5148\u8fdb", "ticker": "688234.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u601d\u745e\u6d66", "short": "\u601d\u745e\u6d66", "ticker": "688536.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u4e1c\u82af\u534a\u5bfc\u4f53", "short": "\u4e1c\u82af\u534a\u5bfc\u4f53", "ticker": "688110.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u701a\u5929\u5929\u6210", "short": "\u701a\u5929\u5929\u6210", "ticker": "2726.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u707f\u52e4\u79d1\u6280", "short": "\u707f\u52e4\u79d1\u6280", "ticker": "688182.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u70ac\u5149\u79d1\u6280", "short": "\u70ac\u5149\u79d1\u6280", "ticker": "688167.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u4e1c\u5fae\u534a\u5bfc", "short": "\u4e1c\u5fae\u534a\u5bfc", "ticker": "688261.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u957f\u5149\u534e\u82af", "short": "\u957f\u5149\u534e\u82af", "ticker": "688048.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u534e\u4e30\u79d1\u6280", "short": "\u534e\u4e30\u79d1\u6280", "ticker": "688629.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u552f\u6377\u521b\u82af", "short": "\u552f\u6377\u521b\u82af", "ticker": "688153.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u6e90\u6770\u79d1\u6280", "short": "\u6e90\u6770\u79d1\u6280", "ticker": "688498.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u5f3a\u4e00\u80a1\u4efd", "short": "\u5f3a\u4e00\u80a1\u4efd", "ticker": "688809.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u6602\u745e\u5fae", "short": "\u6602\u745e\u5fae", "ticker": "688790.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u6770\u534e\u7279", "short": "\u6770\u534e\u7279", "ticker": "688141.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u88d5\u592a\u5fae", "short": "\u88d5\u592a\u5fae", "ticker": "688515.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u7f8e\u82af\u665f", "short": "\u7f8e\u82af\u665f", "ticker": "688458.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u534e\u6d77\u8bda\u79d1", "short": "\u534e\u6d77\u8bda\u79d1", "ticker": "688535.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u5929\u57df\u534a\u5bfc\u4f53", "short": "\u5929\u57df\u534a\u5bfc\u4f53", "ticker": "2658.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u8d5b\u76ee\u79d1\u6280", "short": "\u8d5b\u76ee\u79d1\u6280", "ticker": "2257.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u79d1\u529b\u5c14", "short": "\u79d1\u529b\u5c14", "ticker": "002892.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u62d3\u8346\u79d1\u6280-U", "short": "\u62d3\u8346\u79d1\u6280-U", "ticker": "688072.SS", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u798f\u6676\u79d1\u6280", "short": "\u798f\u6676\u79d1\u6280", "ticker": "002222.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u77fd\u7535\u80a1\u4efd", "short": "\u77fd\u7535\u80a1\u4efd", "ticker": "301629.SZ", "ccy": "CNY", "sleeve": "CN", "status": "active"},
    {"name": "\u601d\u683c\u65b0\u80fd\u6e90", "short": "\u601d\u683c\u65b0\u80fd\u6e90", "ticker": "6656.HK", "ccy": "HKD", "sleeve": "CN", "status": "active"},
    {"name": "\u8d85\u805a\u53d8\u6570\u5b57\u6280\u672f", "short": "\u8d85\u805a\u53d8", "ticker": "SUPERFUSION", "ccy": "CNY", "sleeve": "CN", "status": "prelist", "note": "2026-04-25 IPO tutoring completed; excluded until exchange data exists."},
    {"name": "\u8d5b\u7f8e\u7279", "short": "\u8d5b\u7f8e\u7279", "ticker": "SEMITRONIX", "ccy": "CNY", "sleeve": "CN", "status": "prelist", "note": "Hong Kong IPO push; Hubble holds about 4.96%; excluded until exchange data exists."},
    {"name": "\u5f15\u671b", "short": "\u5f15\u671b", "ticker": "YINWANG", "ccy": "CNY", "sleeve": "CN", "status": "prelist"},
    {"name": "\u9762\u58c1\u667a\u80fd", "short": "\u9762\u58c1\u667a\u80fd", "ticker": "MODELBEST", "ccy": "CNY", "sleeve": "CN", "status": "prelist"},
    {"name": "\u5343\u5bfb\u667a\u80fd", "short": "\u5343\u5bfb\u667a\u80fd", "ticker": "QIANXUNINTELLIGENCE", "ccy": "CNY", "sleeve": "CN", "status": "prelist"},
    {"name": "\u6781\u4f73\u89c6\u754c", "short": "\u6781\u4f73\u89c6\u754c", "ticker": "GEEKSIGHT", "ccy": "CNY", "sleeve": "CN", "status": "prelist"},
    {"name": "iShares 20+ Year Treasury Bond ETF", "short": "TLT", "ticker": "TLT", "ccy": "USD", "sleeve": "US", "status": "active"},
    {"name": "Bitcoin", "short": "BTC", "ticker": "BTC-USD", "ccy": "USD", "sleeve": "CRYPTO", "status": "active"},
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
    "ONDS": 1 / 9,
    "UMAC": 1 / 9,
    "THEON.AS": 0.0,
    "KOPN": 1 / 9,
    "RCAT": 1 / 9,
    "SWMR": 1 / 9,
    "LPTH": 0.0,
    "KTOS": 1 / 9,
    "ESLT": 1 / 9,
    "AVAV": 1 / 9,
    "278A.T": 1 / 9,
    "PLTR": 0.0,
    "RKLB": 0.0,
    "HAWK": 0.0,
    "BKSY": 0.0,
    "SPIR": 0.0,
    "PL": 0.0,
    "LHX": 0.0,
    "02498.HK": 0.0,
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
    "NBIS": 0.0,
    "CRWV": 0.0,
    "IREN": 0.0,
    "APLD": 0.0,
    "WULF": 0.0,
    "HUT": 0.0,
    "CORZ": 0.0,
    "DLR": 0.0,
    "VNET": 0.0,
    "LAMBDA": 0.0,
    "CLICKHOUSE": 0.0,
    "CURSOR": 0.0,
    "LOTUSAI": 0.0,
    "SLINGSHOTAI": 0.0,
    "GETDYNASTY": 0.0,
    "E2B": 0.0,
    "PDD": 0.0,
    "NYT": 0.0,
    "600519.SS": 0.0,
    "000333.SZ": 0.0,
    "0016.HK": 0.0,
    "0001.HK": 0.0,
    "0683.HK": 0.0,
    "2638.HK": 0.0,
    "0003.HK": 0.0,
    "0002.HK": 0.0,
    "2388.HK": 0.0,
    "0005.HK": 0.0,
    "2888.HK": 0.0,
    "300748.SZ": 0.0,
    "688234.SS": 0.0,
    "688536.SS": 0.0,
    "688110.SS": 0.0,
    "2726.HK": 0.0,
    "688182.SS": 0.0,
    "688167.SS": 0.0,
    "688261.SS": 0.0,
    "688048.SS": 0.0,
    "688629.SS": 0.0,
    "688153.SS": 0.0,
    "688498.SS": 0.0,
    "688809.SS": 0.0,
    "688790.SS": 0.0,
    "688141.SS": 0.0,
    "688515.SS": 0.0,
    "688458.SS": 0.0,
    "688535.SS": 0.0,
    "2658.HK": 0.0,
    "2257.HK": 0.0,
    "002892.SZ": 0.0,
    "688072.SS": 0.0,
    "002222.SZ": 0.0,
    "301629.SZ": 0.0,
    "6656.HK": 0.0,
    "SUPERFUSION": 0.0,
    "SEMITRONIX": 0.0,
    "YINWANG": 0.0,
    "MODELBEST": 0.0,
    "QIANXUNINTELLIGENCE": 0.0,
    "GEEKSIGHT": 0.0,
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
    # Lambda Labs is private; keep likely placeholders ready for a future listing.
    "LAMBDA": ["LAMBDA", "LAMBDALABS"],
    # ClickHouse is private; keep likely placeholders ready for a future listing.
    "CLICKHOUSE": ["CLICKHOUSE", "CHDB"],
    # Cursor/Anysphere is private; keep likely placeholders ready for a future listing.
    "CURSOR": ["CURSOR", "ANYS"],
    # Private AI-agent watch names.
    "LOTUSAI": ["LOTUSAI"],
    "SLINGSHOTAI": ["SLINGSHOTAI"],
    "GETDYNASTY": ["GETDYNASTY"],
    "E2B": ["E2B"],
    # Hong Kong exchange codes are sometimes entered with a leading zero in the
    # UI, while Yahoo's chart endpoint uses the four-digit exchange code.
    "02498.HK": ["2498.HK"],
}

AI_CLOUD_ONLY = {
    "NBIS", "CRWV", "IREN", "APLD", "WULF", "HUT", "CORZ", "DLR", "VNET", "LAMBDA", "CLICKHOUSE", "CURSOR",
    "MSFT", "NOW", "LOTUSAI", "SLINGSHOTAI", "GETDYNASTY", "E2B",
}

COMPONENTS = [
    component for component in COMPONENTS
    if component["short"] in AI_CLOUD_ONLY or component["ticker"] in AI_CLOUD_ONLY
]

TARGET_WEIGHTS = {
    key: value for key, value in TARGET_WEIGHTS.items()
    if key in AI_CLOUD_ONLY
}
for ticker in ("NBIS", "CRWV", "IREN", "APLD", "WULF", "HUT", "CORZ", "DLR", "VNET"):
    TARGET_WEIGHTS[ticker] = 1 / 9
TARGET_WEIGHTS["NOW"] = 0.5
TARGET_WEIGHTS["MSFT"] = 0.5
TARGET_WEIGHTS["LAMBDA"] = 0.0
TARGET_WEIGHTS["CLICKHOUSE"] = 0.0
TARGET_WEIGHTS["CURSOR"] = 0.0
TARGET_WEIGHTS["LOTUSAI"] = 0.0
TARGET_WEIGHTS["SLINGSHOTAI"] = 0.0
TARGET_WEIGHTS["GETDYNASTY"] = 0.0
TARGET_WEIGHTS["E2B"] = 0.0

TICKER_CONTINUATIONS = {
    key: value for key, value in TICKER_CONTINUATIONS.items()
    if key in AI_CLOUD_ONLY
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
    df.index = pd.DatetimeIndex(df.index).normalize()
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
        out = out[out.index >= pd.Timestamp(listing_start)]
    return out


def fetch_market_cap(ticker: str):
    try:
        info = yf.Ticker(ticker).fast_info
        market_cap = getattr(info, "market_cap", None)
        if market_cap is None and hasattr(info, "get"):
            market_cap = info.get("market_cap")
        return float(market_cap) if market_cap else None
    except Exception as exc:
        print(f"  WARNING: market cap unavailable for {ticker}: {exc}")
        return None


def fetch_fx(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=False, threads=False)
    if df.empty:
        raise SystemExit(f"FX series unavailable: {ticker}")
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s


def to_records(df: pd.DataFrame) -> list:
    return [
        {"date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
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
    ap.add_argument("--end",   default=(date.today() + timedelta(days=1)).isoformat(),
                    help="Exclusive fetch end date; default is tomorrow so today's closed non-US sessions are included.")
    ap.add_argument("--out",   default=str(script_dir / "data.json"))
    args = ap.parse_args()

    # 1) Fetch all stock OHLC
    fetched = {}
    market_caps_local = {}
    for component in COMPONENTS:
        name, short, ticker, ccy = component["name"], component["short"], component["ticker"], component["ccy"]
        print(f"Fetching {short:15s} ({ticker:12s}, {ccy}) ...")
        fetched[ticker] = fetch_component_ohlc(ticker, args.start, args.end)
        if component.get("status", "active") == "active":
            market_caps_local[ticker] = fetch_market_cap(ticker)

    # 2) Anchor calendar = IREN, preserving the original AI Cloud T0 window.
    anchor_component = next(component for component in COMPONENTS if component["ticker"] == "IREN")
    anchor = fetched[anchor_component["ticker"]]
    if anchor.empty:
        raise SystemExit("Anchor component data missing; cannot establish anchor calendar.")
    common = pd.DatetimeIndex(anchor.index)
    extra_dates = sorted({
        pd.Timestamp(idx)
        for component in COMPONENTS
        for idx in fetched[component["ticker"]].index
        if pd.Timestamp(idx) > common[-1] and pd.Timestamp(idx).weekday() < 5
    })
    if extra_dates:
        common = common.union(pd.Index(extra_dates))
    print(f"Anchor calendar: {len(common)} {anchor_component['short']} trading days "
          f"({common[0]} \u2192 {common[-1]})")

    # 3) Fetch FX, align to anchor calendar
    print("Fetching FX  CNY=X (CNY per USD) ...")
    fx_cny = fetch_fx("CNY=X", args.start, args.end)
    print("Fetching FX  HKD=X (HKD per USD) ...")
    fx_hkd = fetch_fx("HKD=X", args.start, args.end)
    print("Fetching FX  USDEUR=X (EUR per USD) ...")
    fx_eur = fetch_fx("USDEUR=X", args.start, args.end)
    print("Fetching FX  JPY=X (JPY per USD) ...")
    fx_jpy = fetch_fx("JPY=X", args.start, args.end)
    fx_cny = fx_cny.reindex(common).ffill().bfill()
    fx_hkd = fx_hkd.reindex(common).ffill().bfill()
    fx_eur = fx_eur.reindex(common).ffill().bfill()
    fx_jpy = fx_jpy.reindex(common).ffill().bfill()
    latest_fx = {
        "USD": 1.0,
        "CNY": float(fx_cny.dropna().iloc[-1]),
        "HKD": float(fx_hkd.dropna().iloc[-1]),
        "EUR": float(fx_eur.dropna().iloc[-1]),
        "JPY": float(fx_jpy.dropna().iloc[-1]),
    }

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
                "note": component.get("note", "Pre-listing watch code; excluded until exchange data exists."),
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
            "marketCap": ((market_caps_local.get(ticker) or 0.0) / latest_fx.get(ccy, 1.0)) / 1e9,
            "sleeve": component.get("sleeve"),
            "status": status,
            "inception": pd.Timestamp(inception).strftime("%Y-%m-%d") if inception is not None else None,
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
            "fx_unit_JPY": "JPY per USD (yfinance: JPY=X)",
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
            "CNY": [{"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "rate": float(r)} for idx, r in fx_cny.items()],
            "HKD": [{"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "rate": float(r)} for idx, r in fx_hkd.items()],
            "EUR": [{"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "rate": float(r)} for idx, r in fx_eur.items()],
            "JPY": [{"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "rate": float(r)} for idx, r in fx_jpy.items()],
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
