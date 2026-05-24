import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-nelson-siegel-results"

# Universes updated to match actual master data columns
FI_TICKERS        = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "SLV", "GLD"]
EQUITY_TICKERS    = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "GDX", "XME", "IWM", "IWF", "XSD", "XBI",
    "XLB", "XLRE", "IWD", "IWO",
]
COMBINED_TICKERS  = FI_TICKERS + EQUITY_TICKERS

UNIVERSES = {
    "FI_COMMODITIES": FI_TICKERS,
    "EQUITY_SECTORS": EQUITY_TICKERS,
    "COMBINED":       COMBINED_TICKERS,
}

# Treasury columns — will be fetched by update_master_data.py and added to parquet
# These match the FRED series names used in update_master_data.py
TREASURY_COLS = {
    "DGS1MO": 1/12,
    "DGS3MO": 3/12,
    "DGS6MO": 6/12,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS5":   5.0,
    "DGS7":   7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}

# Rolling windows for NS parameter estimation
WINDOWS = [63, 126, 252, 504]

# NS signal construction
LAMBDA_INIT       = 0.7    # initial decay parameter for NS fitting
NS_SCORE_LOOKBACK = 21     # days to compute recent factor change

TOP_N = 3
