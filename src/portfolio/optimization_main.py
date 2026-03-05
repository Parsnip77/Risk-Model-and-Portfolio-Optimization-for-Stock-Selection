"""
optimization_main.py
--------------------
Stage 4 entry script: convex portfolio optimisation backtest.

Workflow
--------
1.  Load ``data/ml_alpha.parquet`` (produced by ml_analyze_main.py),
    ``data/prices.parquet``, and ``data/meta.parquet``.
2.  Instantiate ``OptimizationBacktester`` with the industry-neutral,
    cost-aware convex optimiser (PortfolioOptimizer).
3.  Run the backtest: for each trading day, the optimiser solves the LP

        max  w' alpha_centered - lambda_turnover/2 * ||w - w_prev||_1
        s.t. sum(w)=1, w>=0, w<=max_weight,
             |X_ind' w - w_bench| <= industry_tol  (auto-relaxed if needed)

    where alpha_centered is the cross-sectionally de-meaned ML alpha.
    Gross returns, one-way turnover, and net returns are then accumulated.
    Net return deducts cost_rate * turnover per day for P&L accounting.

4.  Print and save a performance summary to ``result_optimization.txt``.
5.  Save cumulative NAV chart to ``plots/optimization_nav.png``.
6.  Print a brief log of industry-tolerance relaxation events.

Key parameters (see Configuration section below)
-------------------------------------------------
  LAMBDA_TURNOVER
      Dimensionless turnover-aversion coefficient in the LP objective.
      Controls signal-vs-stability trade-off; calibrate by inspecting
      ``Avg Daily Turnover`` in the output report.  NOT a cost rate.
      Suggested range: 0.05 (aggressive) to 1.0 (very stable).

  COST_RATE
      Actual one-way transaction cost rate for P&L accounting only
      (default 0.002 = 0.2 %).  Deducted as turnover * COST_RATE per day.
      Separate from LAMBDA_TURNOVER; do not conflate the two.

Prerequisites
-------------
Run ``src/LightGBM/ml_analyze_main.py`` first to generate ``ml_alpha.parquet``.

Usage
-----
    python src/portfolio/optimization_main.py
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
import warnings
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# __file__ is src/portfolio/optimization_main.py
# → project root is two levels up
_ROOT       = pathlib.Path(__file__).parent.parent.parent
_PORT_DIR   = pathlib.Path(__file__).parent          # src/portfolio/
sys.path.insert(0, str(_PORT_DIR))                   # for optimizer, optimization_backtester

warnings.filterwarnings("ignore")

from optimization_backtester import OptimizationBacktester  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR    = _ROOT / "data"
PLOTS_DIR   = _ROOT / "plots"
RESULT_FILE = _ROOT / "result_optimization.txt"

# --- Two separate cost-related parameters (do NOT merge into one) -----------

LAMBDA_TURNOVER: float = 0.015
# Turnover-aversion coefficient in the LP objective.  Dimensionless policy knob.
# Because alpha is in rank-score units (~±0.5 after de-meaning), NOT return
# units (~±0.01), using the monetary cost rate (0.002) here would make the
# penalty ~250x too small to matter.  Tune empirically:
#   0.05~0.1  → aggressive, Avg Daily Turnover ~10–20 %
#   0.2~0.5   → moderate,   Avg Daily Turnover ~2–8 %  ← recommended start
#   1.0+      → very stable, slow signal tracking

COST_RATE: float = 0.002
# Actual one-way transaction cost rate for P&L deduction only.
# Applied as: net_return = gross_return - turnover * COST_RATE.
# Do not use this as the optimiser penalty; see LAMBDA_TURNOVER above.

# --- Other optimiser settings -----------------------------------------------
RF:             float = 0.03     # annual risk-free rate
MAX_WEIGHT:     float = 0.05     # per-stock weight cap
INDUSTRY_TOL:   float = 0.01     # initial industry deviation tolerance (±1 pp)

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print(msg: str, buf: StringIO) -> None:
    print(msg)
    buf.write(msg + "\n")


def _section(title: str, buf: StringIO, width: int = 70) -> None:
    _print("\n" + "=" * width, buf)
    _print(f"  {title}", buf)
    _print("=" * width, buf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    report_buf = StringIO()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    _section("Stage 4 · Convex Portfolio Optimisation Backtest", report_buf)

    for name, path in [
        ("ml_alpha",  DATA_DIR / "ml_alpha.parquet"),
        ("prices",    DATA_DIR / "prices.parquet"),
        ("meta",      DATA_DIR / "meta.parquet"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                + (
                    "Run src/LightGBM/ml_analyze_main.py first to generate ml_alpha.parquet."
                    if name == "ml_alpha"
                    else "Run src/data_preparation/data_preparation_main.py first."
                )
            )

    _print("1. Loading parquet files ...", report_buf)
    alpha_df  = pd.read_parquet(DATA_DIR / "ml_alpha.parquet")
    prices_df = pd.read_parquet(DATA_DIR / "prices.parquet")
    meta_df   = pd.read_parquet(DATA_DIR / "meta.parquet")

    # Flatten MultiIndex if present
    for df in (alpha_df, prices_df, meta_df):
        if isinstance(df.index, pd.MultiIndex):
            df.reset_index(inplace=True)

    # CSI 300 index prices for benchmark comparison (optional: skip if file missing)
    index_path = DATA_DIR / "index.parquet"
    index_prices: pd.Series | None = None
    if index_path.exists():
        index_prices = (
            pd.read_parquet(index_path)
            .set_index("trade_date")["close"]
        )

    _print(f"   ml_alpha shape : {alpha_df.shape}", report_buf)
    _print(f"   prices   shape : {prices_df.shape}", report_buf)
    _print(f"   meta     shape : {meta_df.shape}", report_buf)
    _print(
        f"   index prices  : {len(index_prices)} rows"
        if index_prices is not None else "   index prices  : not found (no benchmark comparison)",
        report_buf,
    )
    _print(
        f"   alpha date range : {alpha_df['trade_date'].min()} → "
        f"{alpha_df['trade_date'].max()}",
        report_buf,
    )

    # -----------------------------------------------------------------------
    # 2. Optimisation settings summary
    # -----------------------------------------------------------------------
    _section("Optimisation Settings", report_buf)
    _print(
        textwrap.dedent(f"""
        Objective  : max  w' alpha_centered - lambda_TO/2 * ||w - w_prev||_1
                     (alpha_centered = alpha - cross_sectional_mean(alpha))

        lambda_turnover : {LAMBDA_TURNOVER}
          → turnover-aversion coefficient (dimensionless policy knob, NOT a cost rate)
          → tune by inspecting Avg Daily Turnover in the report below

        cost_rate : {COST_RATE:.4f}
          → actual one-way transaction cost rate for P&L accounting only
          → net_return = gross_return - turnover * cost_rate

        Max weight per stock : {MAX_WEIGHT:.0%}
        Industry neutrality  : ±{INDUSTRY_TOL:.0%} tolerance (auto-relax up to ±5%)
        Solver               : CLARABEL (cvxpy default for LP)
        """).strip(),
        report_buf,
    )

    # -----------------------------------------------------------------------
    # 3. Run backtest
    # -----------------------------------------------------------------------
    _section("Running Optimisation Backtest ...", report_buf)
    _print("(This may take several minutes — one LP solve per trading day.)", report_buf)

    obt = OptimizationBacktester(
        alpha_df,
        prices_df,
        meta_df,
        cost_rate=COST_RATE,              # P&L cost deduction rate
        lambda_turnover=LAMBDA_TURNOVER,  # optimiser penalty coefficient
        rf=RF,
        max_weight=MAX_WEIGHT,
        industry_tol=INDUSTRY_TOL,
        plots_dir=PLOTS_DIR,
        benchmark_prices=index_prices,    # CSI 300 for excess-return metrics
    )

    summary = obt.run_backtest()

    # -----------------------------------------------------------------------
    # 4. Performance report
    # -----------------------------------------------------------------------
    _section("Performance Summary", report_buf)
    _print("\n" + summary.to_string(), report_buf)

    # -----------------------------------------------------------------------
    # 5. Industry tolerance relaxation log
    # -----------------------------------------------------------------------
    _section("Industry Tolerance Relaxation Log", report_buf)

    relax_events = [
        e for e in obt._relax_log if "relaxed" in e.get("message", "")
    ]
    n_relax = len(relax_events)
    n_days  = len(obt._net_ret) if obt._net_ret is not None else 0

    _print(
        f"   Relaxation events : {n_relax} / {n_days} trading days "
        f"({n_relax / n_days * 100:.1f}%)" if n_days > 0 else
        "   No backtest data available.",
        report_buf,
    )
    if relax_events:
        _print("   First 10 events:", report_buf)
        for e in relax_events[:10]:
            _print(f"     {e['date']}  —  {e['message']}", report_buf)
        if n_relax > 10:
            _print(f"   ... and {n_relax - 10} more.", report_buf)

    # -----------------------------------------------------------------------
    # 6. Save chart and report
    # -----------------------------------------------------------------------
    _section("Saving Outputs", report_buf)

    obt.plot(show=False)
    nav_path = PLOTS_DIR / "optimization_nav.png"
    _print(f"   NAV chart saved  : {nav_path}", report_buf)

    report_text = report_buf.getvalue()
    RESULT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to : {RESULT_FILE}")
    print("Chart saved to       : plots/optimization_nav.png")


if __name__ == "__main__":
    main()
