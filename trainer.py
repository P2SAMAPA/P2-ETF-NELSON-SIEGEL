import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import config
import data_manager
from nelson_siegel import (
    extract_ns_factors,
    compute_etf_ns_betas,
    compute_ns_scores,
)


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df    = data_manager.load_master_data()
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Print all columns so we can see what's available ─────────────────────
    print(f"\nMaster data shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # ── Auto-detect treasury columns ──────────────────────────────────────────
    print("\n=== Auto-detecting treasury yield columns ===")
    treasury_cols = data_manager.detect_treasury_cols(df)

    if len(treasury_cols) < 4:
        print(f"ERROR: Only {len(treasury_cols)} treasury columns found — need at least 4.")
        print("Available columns in master data:")
        for col in sorted(df.columns):
            sample = df[col].dropna()
            if len(sample) > 0:
                print(f"  {col}: median={sample.median():.4f}, "
                      f"n={len(sample)}, dtype={df[col].dtype}")
        return

    # ── Extract NS factors once (shared across all universes / windows) ────────
    print("\n=== Fitting Nelson-Siegel yield curve ===")
    ns_factors_by_window = {}
    for win in config.WINDOWS:
        print(f"  Fitting NS factors — window {win}d...")
        factors = extract_ns_factors(
            df,
            treasury_cols,       # use auto-detected cols, not config hardcoded
            window=win,
            lambda_init=config.LAMBDA_INIT,
        )
        if factors.empty:
            print(f"  No NS factors for window {win}d, skipping")
            continue
        ns_factors_by_window[win] = factors
        last = factors.iloc[-1]
        print(f"  Done — {len(factors)} obs · "
              f"β₀={last['beta0']:.4f} β₁={last['beta1']:.4f} "
              f"β₂={last['beta2']:.4f} λ={last['lam']:.4f} RMSE={last['rmse']:.5f}")

    if not ns_factors_by_window:
        print("No NS factors extracted — exiting")
        return

    all_results = {}
    all_windows = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Nelson-Siegel) ===")

        returns = data_manager.prepare_returns_matrix(df, tickers)
        available_tickers = [t for t in tickers if t in returns.columns]

        if not available_tickers or returns.empty:
            print("  No data available")
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        best_per_etf   = {}
        window_results = {}

        for win in config.WINDOWS:
            if win not in ns_factors_by_window:
                continue
            if len(returns) < win + config.NS_SCORE_LOOKBACK + 5:
                print(f"  Skipping window {win}d (insufficient data)")
                continue

            print(f"  Processing window {win}d...")
            factors = ns_factors_by_window[win]

            try:
                ns_betas = compute_etf_ns_betas(
                    returns[available_tickers], factors, window=win)
            except Exception as e:
                print(f"  Beta computation failed for {win}d: {e}")
                continue

            if ns_betas.empty:
                print(f"  No betas for {win}d")
                continue

            try:
                scores = compute_ns_scores(
                    returns[available_tickers], factors, ns_betas,
                    score_lookback=config.NS_SCORE_LOOKBACK)
            except Exception as e:
                print(f"  Score computation failed for {win}d: {e}")
                continue

            if scores.empty:
                print(f"  No scores for {win}d")
                continue

            score_dict = scores.to_dict()
            window_results[win] = {t: float(s) for t, s in score_dict.items()}
            top = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top 3: {[(t, round(s,4)) for t, s in top]}")

            for etf, score in score_dict.items():
                if np.isnan(score):
                    continue
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (float(score), win)

        # ── Fallback ──────────────────────────────────────────────────────────
        if not best_per_etf:
            print("  Falling back to historical mean return")
            for etf in available_tickers:
                mean_ret = returns[etf].iloc[-252:].mean()
                if not np.isnan(mean_ret):
                    best_per_etf[etf] = (float(mean_ret), 0)

        if not best_per_etf:
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        # ── Tab 1 ─────────────────────────────────────────────────────────────
        full_scores = {
            ticker: {"score": float(score), "best_window": int(win)}
            for ticker, (score, win) in best_per_etf.items()
        }
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs    = [
            {"ticker": t, "ns_score": float(s), "best_window": int(w)}
            for t, (s, w) in sorted_etfs[:config.TOP_N]
        ]
        print(f"  Top {config.TOP_N}: {[e['ticker'] for e in top_etfs]}")

        all_results[universe_name] = {
            "top_etfs": top_etfs, "full_scores": full_scores,
            "window_results": window_results, "run_date": today,
        }

        # ── Tab 2 ─────────────────────────────────────────────────────────────
        windows_tab2 = {}
        for win, sd in window_results.items():
            sw = sorted(sd.items(), key=lambda x: x[1], reverse=True)
            windows_tab2[str(win)] = {
                "top_etfs":    [{"ticker": t, "ns_score": float(s)} for t, s in sw[:config.TOP_N]],
                "full_ranking":[{"ticker": t, "ns_score": float(s)} for t, s in sw],
            }
        all_windows[universe_name] = {"windows": windows_tab2, "run_date": today}

    # ── NS factor summary for app display ────────────────────────────────────
    ns_summary = {}
    for win, factors in ns_factors_by_window.items():
        last = factors.iloc[-1]
        ns_summary[str(win)] = {
            "date":  str(factors.index[-1].date()),
            "beta0": float(last["beta0"]),
            "beta1": float(last["beta1"]),
            "beta2": float(last["beta2"]),
            "lam":   float(last["lam"]),
            "rmse":  float(last["rmse"]),
        }

    # ── Write and push ────────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)

    tab1_path = Path(f"results/nelson_siegel_{today}.json")
    with open(tab1_path, "w") as f:
        json.dump(convert_to_serializable({
            "run_date": today, "ns_factors": ns_summary, "universes": all_results,
        }), f, indent=2)

    tab2_path = Path(f"results/nelson_siegel_windows_{today}.json")
    with open(tab2_path, "w") as f:
        json.dump(convert_to_serializable({
            "run_date": today, "universes": all_windows,
        }), f, indent=2)

    import push_results
    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)

    print(f"\n=== Nelson-Siegel Engine complete ===")
    print(f"  Tab 1: {tab1_path.name}")
    print(f"  Tab 2: {tab2_path.name}")


if __name__ == "__main__":
    main()
