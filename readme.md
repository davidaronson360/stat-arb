# **Cointegration-Based Stat-Arb — Cross-Validated, Cost-Aware Execution**

> **Objective:** Document a reproducible, cost-aware cointegration stat-arb pipeline with 5-fold cross-validation, T+1 execution, and transparent net/excess reporting.

---

## Executive Summary — Results & Methodological Distinctions

- **Formation → Gating → Trading → P&L** pipeline with explicit audit points.
- **Pairs and baskets** within industries via **Engle–Granger (EG)** and **Johansen** on **log prices** (default; configurable to levels).
- **Robust Z‑scores** (rolling **median/MAD**) for entry/exit to reduce outlier sensitivity.
- **Stationarity & mean‑reversion checks:**
  - **ADF** (p < α) with **negative adjustment slope**
  - **AR(1)**: 0 < ρ < 1 and **HAC‑robust** significance
  - **OU half‑life** derived from AR(1); **time‑stop = 4×HL (selected via 5‑fold CV)**
- **Multiple‑testing control:** **Benjamini–Hochberg (BH‑FDR)** over ADF p‑values per formation window.
- **Execution discipline:** **no look‑ahead**; indicators use only past data; **T+1 order placement** option to avoid same‑bar bias.
- **Costs on every trade:** commission, half‑spread ticks, and slippage (bps) with sensitivity analysis.
- **Risk management:** risk‑parity sizing on ΔSpread, portfolio vol‑targeting, hard caps on # active spreads and gross leverage, dollar‑neutral legs.
- **Model selection by out‑of‑sample logic:** 5‑fold CV with non‑overlapping 1‑yr trading windows (2017–2018, …, 2022–2023). Optimized hyperparameters included: ADF p‑value threshold, BH‑FDR α value;  entry/exit Z bands; hard‑stop Z; time‑stop multiplier (CV‑selected 4 × Halflife);  max concurrent spreads. Selection criterion: average Net Sharpe across folds.

**Headline metrics**:

- **Out‑of‑sample (selected by CV):** 2024‑01‑01 → 2025‑09‑27\
  **Net Sharpe:** 0.77 · **Excess Net Sharpe** (vs \~4% RF): **−0.47**\
  Trades: 26 · Net APR: 2.43% · Net P&L: 42,101.67

- **Full period (not OOS for hyperparams):** 2016‑01‑01 → 2025‑09‑27\
  **Net Sharpe:** 0.43 · **Excess Net Sharpe:** −0.62\
  Steps: 105 · Trades: 255 · Net APR: 1.60% · Net P&L: 155,020.37

Exit‑reason mix (OOS 2024–2025): time\_stop 46.2%, band 30.8%, end\_of\_window 11.5%, hard\_stop 11.5%.

---

## Methodology & Implementation — Design Rationale → Code

**Formation** (`src/formation.py`)

- Scans **within‑industry** universes (Energy, Airlines, Consumer Staples—Beverages, Banks, Utilities) to keep economic comparability.
- **EG pairs** and **Johansen baskets** on **log‑price panels**. Returns immutable **SpreadSpec** objects with diagnostics for later gating.

**Gating** (`src/gating.py`)

- **ADF test** with both **p‑value threshold** and **negative adjustment slope** check.
- **AR(1) fit**; require 0 < ρ < 1 and significance under **HAC (Newey–West) standard errors** (heteroskedasticity & autocorrelation consistent), using an automatic lag choice maxlags ≈ 1.5·√n.
- **Half‑life** from AR(1) → **time‑stop = 4×HL** (configurable bounds on HL).
- **BH‑FDR** applied across simultaneous ADF tests to control expected false discoveries.

**Signal & OU utilities** (`src/ou.py`)

- **Rolling robust Z** via **median/MAD** to temper outliers.
- **AR(1)↔OU mapping** to compute mean‑reversion **κ** and **half‑life**.

**Portfolio sizing & risk** (`src/portfolio.py`)

- Size on **absolute ΔSpread**, not percent returns, matching the stationary object being traded.
- **Risk‑parity:** weights ∝ 1/σ\_i; scale to a **target daily/annual vol**, assuming weak cross‑spread correlation.
- **Dollar‑neutral** leg splitting by |weights| at current prices; hard caps on **# active spreads** and **gross leverage**.

**Costs** (`src/costs.py`)

- **Commission per share**, **1 half‑spread tick** per entry/exit leg, **slippage in bps**; bundled into every trade.
- **Sensitivity** helpers to ±50% slippage for robustness.

**Backtest engine** (`src/backtest.py`)

- Strict **indicator alignment** (no forward‑fill past the bar being decided).
- Optional **T+1 execution** (generate at t, act at t+1) to eliminate same‑bar bias.
- Exit priority: **band** → **hard\_stop** → **time\_stop** → **end\_of\_window**; optional **min‑hold** to avoid 1‑day churn.
- Verbose accounting of **P&L**, **costs**, and **exit reasons**.

**Data & universe** (`src/data.py`, `src/universe.py`)

- Adjusted closes (splits/dividends) with deterministic local **Parquet cache**; alignment to an **M–F daily calendar**, pruning all‑NA days.
- Guardrails on missingness (forward‑fill caps), liquid tickers only.

---

## Cross‑Validation Protocol (what “5‑fold” means here)

- **Five overlapping 2‑year folds (rolling by one year) with non‑overlapping 1‑yr trading windows**:\
  (1) 2017–2018, (2) 2018–2019, (3) 2019–2020, (4) 2020–2021, (5) 2022–2023\
  (each with a 1‑year learn + 1‑year trade cadence inside the fold, enforced in code).
- For each candidate parameter row, run the **entire pipeline** on each fold.
- Score by **average Net Sharpe** across the 5 folds.
- **Lock** the best row; **do not touch** it while evaluating **out‑of‑sample** (2024‑01‑01 onward).

> This structure, combined with BH‑FDR and T+1, is intentionally conservative. It prefers stability of effect over isolated good years.

---

## Detailed metrics (complement to headline numbers)

**Only items not shown above**:

### Full period (2016‑01‑01 → 2025‑09‑27)

**Gross APR:** 1.98% · **Gross Sharpe:** 0.53 · **Gross P&L:** 192,017.69  
**Exit reasons:** time\_stop 46.7% · band 31.8% · hard\_stop 12.2% · end\_of\_window 9.4%

### Out‑of‑sample (selected by 5‑fold CV; 2024‑01‑01 → 2025‑09‑27)

**Gross APR:** 2.67% · **Gross Sharpe:** 0.84 · **Gross P&L:** 46,140.33

*Excess Sharpe computed against \~4% RF.*

---

## Key takeaways for reviewers

1. **Methodology first.** Every stage is explicitly tested or regularized (ADF/AR, OU half‑life, BH‑FDR, robust Z, T+1, costs).
2. **Transparent trade‑offs.** Results are presented **net of costs**; excess Sharpe is shown against a realistic risk-free rate.
3. **Reproducible.** Parameter grids, CV folds, and OOS periods are **code‑enforced** and documented.
4. **Risk managed.** Sizing on ΔSpread with vol targeting and leverage/gross caps is appropriate to stationary spread trading.

---

## Limitations & future work&#x20;

- **Universe breadth.** Currently focused on 5 industries; expanding and adding **liquidity/borrow screens** may improve capacity and realism.
- **Hedge‑ratio dynamics.** **Kalman filter / RLS** used to update the hedge ratio **β\_t** continuously (time‑varying).&#x20;
- **Window continuity.** **Trades may continue across formation/CV windows** (no forced flattening at boundaries). Future work: quantify boundary effects and add regime‑break guardrails.
- **Regime awareness.** Consider time‑varying gates (e.g., dynamic FDR α, HL bounds) and **robust regression** for hedge‑ratios under volatility regimes.

---

## Repo guide

- `src/formation.py` — propose EG pairs & Johansen baskets; industry‑scoped.
- `src/gating.py` — ADF/AR(1) with HAC; OU half‑life; BH‑FDR; return tradable spreads + time‑stops.
- `src/ou.py` — robust Z (median/MAD); AR(1)↔OU maps; half‑life utilities.
- `src/portfolio.py` — position sizing on ΔSpread; vol targeting; gross/#spread caps; dollar‑neutral legs.
- `src/costs.py` — commission, half‑spread ticks, slippage bps; sensitivity helpers.
- `src/backtest.py` — end‑to‑end engine; T+1 option; exit logic; P&L & audit tables.
- `src/data.py` — adjusted prices + deterministic Parquet cache; M–F calendar alignment.
- `src/universe.py` — parse/validate `configs/base.yaml` (industries, dates).
- `src/generate_zoom_params_v6.py` — create CV parameter CSV (incl. ADF α sweep).
- `src/run_kfold_cv_v6.py` — run 5‑fold CV and aggregate results.

