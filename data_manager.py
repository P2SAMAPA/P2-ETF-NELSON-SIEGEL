import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config


def load_master_data() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=config.DATA_REPO,
        filename="master_data.parquet",
        repo_type="dataset",
        token=config.HF_TOKEN,
    )
    df = pd.read_parquet(path)
    if df.index.name != "date":
        df.index.name = "date"
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def detect_treasury_cols(df: pd.DataFrame) -> dict:
    """
    Auto-detect treasury yield columns in master data.

    Tries multiple common naming conventions from FRED, Bloomberg,
    and P2SAMAPA master data. Returns a dict of {col_name: maturity_years}
    for every column found in the DataFrame.

    Falls back gracefully — the NS model needs at least 4 maturities.
    """
    # All known naming conventions for US Treasury yields
    candidates = [
        # FRED standard names
        ("DGS1MO",  1/12),
        ("DGS3MO",  3/12),
        ("DGS6MO",  6/12),
        ("DGS1",    1.0),
        ("DGS2",    2.0),
        ("DGS3",    3.0),
        ("DGS5",    5.0),
        ("DGS7",    7.0),
        ("DGS10",   10.0),
        ("DGS20",   20.0),
        ("DGS30",   30.0),
        # Alternative FRED names
        ("GS1M",    1/12),
        ("GS3M",    3/12),
        ("GS6M",    6/12),
        ("GS1",     1.0),
        ("GS2",     2.0),
        ("GS3",     3.0),
        ("GS5",     5.0),
        ("GS7",     7.0),
        ("GS10",    10.0),
        ("GS20",    20.0),
        ("GS30",    30.0),
        # P2SAMAPA / yfinance naming conventions
        ("T1M",     1/12),
        ("T3M",     3/12),
        ("T6M",     6/12),
        ("T1Y",     1.0),
        ("T2Y",     2.0),
        ("T3Y",     3.0),
        ("T5Y",     5.0),
        ("T7Y",     7.0),
        ("T10Y",    10.0),
        ("T20Y",    20.0),
        ("T30Y",    30.0),
        # With underscore prefix
        ("_1M",     1/12),
        ("_3M",     3/12),
        ("_6M",     6/12),
        ("_1Y",     1.0),
        ("_2Y",     2.0),
        ("_5Y",     5.0),
        ("_10Y",    10.0),
        ("_30Y",    30.0),
        # Yield_ prefix
        ("YIELD_1M",  1/12),
        ("YIELD_3M",  3/12),
        ("YIELD_6M",  6/12),
        ("YIELD_1Y",  1.0),
        ("YIELD_2Y",  2.0),
        ("YIELD_5Y",  5.0),
        ("YIELD_10Y", 10.0),
        ("YIELD_30Y", 30.0),
        # Rate_ prefix
        ("RATE_1M",  1/12),
        ("RATE_3M",  3/12),
        ("RATE_1Y",  1.0),
        ("RATE_2Y",  2.0),
        ("RATE_5Y",  5.0),
        ("RATE_10Y", 10.0),
        ("RATE_30Y", 30.0),
        # DTB (discount/T-bill) series
        ("DTB4WK",  1/12),
        ("DTB3",    3/12),
        ("DTB6",    6/12),
    ]

    found = {}
    for col, mat in candidates:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 100:   # must have meaningful history
                found[col] = mat

    # Also do a fuzzy scan: any column whose name contains a maturity keyword
    # and has values in the range 0–25 (percent yield range)
    maturity_keywords = {
        "1M": 1/12, "3M": 3/12, "6M": 6/12,
        "1Y": 1.0,  "2Y": 2.0,  "3Y": 3.0,
        "5Y": 5.0,  "7Y": 7.0,  "10Y": 10.0,
        "20Y": 20.0, "30Y": 30.0,
    }
    for col in df.columns:
        if col in found:
            continue
        col_upper = col.upper()
        for kw, mat in maturity_keywords.items():
            if kw in col_upper:
                series = df[col].dropna()
                if len(series) > 100:
                    median_val = series.median()
                    # Yields should be between 0 and 25 percent
                    if 0.0 < median_val < 25.0:
                        found[col] = mat
                        break

    print(f"  Auto-detected {len(found)} treasury columns: {list(found.keys())}")
    return found


def prepare_returns_matrix(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    returns = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    return returns.dropna(how="all")
