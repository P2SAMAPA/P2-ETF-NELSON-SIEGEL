# P2-ETF-NELSON-SIEGEL

**Nelson-Siegel Yield Curve Engine** — part of the P2Quant Engine Suite (v14).

Fits the Nelson-Siegel (1987) parametric yield curve model to the US Treasury term structure daily, extracts three latent factors (Level, Slope, Curvature), and ranks ETFs by their sensitivity to factor changes using the Diebold-Li (2006) dynamic factor approach.

---

## Mathematical Foundation

### Nelson-Siegel Model (1987)

```
y(τ) = β₀
      + β₁ · [(1 − exp(−λτ)) / (λτ)]
      + β₂ · [(1 − exp(−λτ)) / (λτ) − exp(−λτ)]
```

| Parameter | Economic Interpretation | ETF Impact |
|-----------|------------------------|------------|
| β₀ Level | Long-run yield level | Rising = bad for duration (TLT, LQD) |
| β₁ Slope | Short vs long rate spread | Steepening = risk-on (HYG, credit ETFs) |
| β₂ Curvature | Medium-term hump | Hump = pressure on mid-duration ETFs |
| λ Decay | Hump location (optimised daily) | Controls NS fit quality |

### Diebold-Li (2006) Dynamic Factor Model

β₀, β₁, β₂ are treated as time-varying state variables estimated daily via OLS on the observed Treasury cross-section.

### ETF Sensitivity Betas

Rolling OLS of each ETF's daily return on daily changes in NS factors:
```
r_i(t) = α_i + b0_i·Δβ₀(t) + b1_i·Δβ₁(t) + b2_i·Δβ₂(t) + ε_i(t)
```

### Composite Score

```
score_i = −w_level·(b0_i × Δβ₀_recent)
          + w_slope·(b1_i × Δβ₁_recent)
          + w_curve·(b2_i × Δβ₂_recent)
```

Cross-sectionally z-scored. Weights: Level 40% | Slope 35% | Curvature 25%.

---

## Repository Structure

```
P2-ETF-NELSON-SIEGEL/
├── config.py             # Universe, windows, weights
├── nelson_siegel.py      # NS fitting, beta computation, scoring
├── data_manager.py       # HF data loader + auto-detection of treasury columns
├── trainer.py            # Main runner
├── streamlit_app.py      # Two-tab Streamlit dashboard
├── push_results.py       # HuggingFace upload
├── us_calendar.py        # Next trading day
├── requirements.txt
└── .github/workflows/daily.yml
```

---

## Treasury Column Auto-Detection

`data_manager.detect_treasury_cols()` automatically scans the master data parquet for any treasury yield columns, trying all common naming conventions:

- FRED standard: `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `DGS5`, `DGS7`, `DGS10`, `DGS20`, `DGS30`
- Alternative FRED: `GS1M`, `GS3M`, `GS1`, `GS2`, `GS5`, `GS10`, `GS30`
- P2SAMAPA format: `T1M`, `T3M`, `T1Y`, `T2Y`, `T5Y`, `T10Y`, `T30Y`
- Keyword fuzzy scan: any column containing `1Y`, `2Y`, `5Y`, `10Y` etc with values in 0–25% range

The trainer also prints **all column names** in the master data on first run so you can identify the exact names if auto-detection misses any.

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO |
| COMBINED | All of the above |

---

## Rolling Windows

| Window | Duration | Recommendation |
|--------|----------|---------------|
| 63d | ~3 months | Short-term rate sensitivity |
| 126d | ~6 months | Medium-term |
| 252d | ~1 year | **Recommended** — stable betas |
| 504d | ~2 years | Long-run structural sensitivity |

---

## Output Files (pushed to HuggingFace)

| File | Tab | Content |
|------|-----|---------|
| `nelson_siegel_YYYY-MM-DD.json` | Tab 1 | Best window per ETF + NS factor values |
| `nelson_siegel_windows_YYYY-MM-DD.json` | Tab 2 | Full ranking at every window |

---

## Streamlit App

**Tab 1 — Best Window per ETF**
- Current NS factor values (β₀, β₁, β₂, λ, RMSE) per window
- Top 3 ETF cards per universe
- Full ranking table

**Tab 2 — Explore by Window**
- Window dropdown (63d / 126d / 252d / 504d)
- Top 3 cards + full ranking per universe at selected window

---

## Setup

1. Create GitHub repo `P2-ETF-NELSON-SIEGEL`
2. Create HuggingFace dataset `P2SAMAPA/p2-etf-nelson-siegel-results`
3. Add `HF_TOKEN` as a GitHub Actions secret
4. In repo Settings → Actions → General → set **Workflow permissions** to **Read and write**
5. Push all files to `main`
6. Actions → Run workflow

---

## References

- Nelson, C., Siegel, A. (1987). *Parsimonious Modeling of Yield Curves*. Journal of Business.
- Diebold, F., Li, C. (2006). *Forecasting the Term Structure of Government Bond Yields*. Journal of Econometrics.

**HuggingFace Results:** `P2SAMAPA/p2-etf-nelson-siegel-results`  
**Part of:** P2Quant Engine Suite · P2SAMAPA
