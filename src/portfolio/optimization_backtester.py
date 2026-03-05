"""
optimization_backtester.py
--------------------------
Long-only backtester that uses a convex portfolio optimiser to determine daily
holdings, replacing the simple quantile-based selection used in net_backtester.py.

Each trading day the backtester calls ``PortfolioOptimizer.solve()`` to obtain
the optimal weight vector w_t, then accounts for transaction costs and computes
net returns.

Two-parameter design (important)
----------------------------------
This class uses TWO separate cost-related parameters with distinct roles:

  lambda_turnover (default 0.2)
      Turnover-aversion coefficient in the optimiser objective.  Controls how
      aggressively the optimiser chases the alpha signal versus maintaining
      stable positions.  This is a *policy knob*, not a monetary rate.

      Because alpha_t is in cross-sectional rank-score units (~±0.5 after
      de-meaning) rather than return units (~±0.01), using the actual cost
      rate (0.002) as this coefficient would make the penalty term ~250x
      smaller than the alpha term, rendering it effectively zero.

  cost_rate (default 0.002)
      The actual one-way transaction cost rate used for P&L accounting.
      Applied as:  net_return = gross_return - turnover * cost_rate.
      This always reflects real-world trading friction and must not be
      conflated with lambda_turnover.

Alpha normalisation
--------------------
Before passing alpha to the optimiser, the cross-sectional mean is subtracted
on each trading day so that the signal is centred around 0:

    alpha_centered = alpha_t - mean(alpha_t)

This de-meaning step:
  (a) makes the signal symmetric (top stocks get positive values, bottom
      stocks get negative values, consistent with a long-only tilt away
      from zero);
  (b) ensures lambda_turnover operates in a stable numeric range regardless
      of the raw prediction scale;
  (c) does NOT change the cross-sectional ranking, so IC and all ordering-
      based metrics remain unaffected.

Execution assumption (open-to-open, no look-ahead bias)
--------------------------------------------------------
Signal is computed from data available at the close of day T.  The trade
executes at the OPEN of day T+1 and is held until the OPEN of day T+2.
Realised portfolio return each period is therefore the open-to-open return:

    stock_ret[T] = open_T / open_{T-1} - 1

This is consistent with ``calc_forward_return`` in targets.py, which sets
    forward_return_T = open_{T+2} / open_{T+1} - 1
so the LP objective (maximise expected forward return) and the P&L
accounting use identical return definitions.

Portfolio construction (per day t)
------------------------------------
1. Retrieve alpha_t (ML predictions) and tradability flag for each stock.
2. Filter to stocks in S_t: tradable and with a valid (non-NaN) alpha.
3. Compute benchmark industry weights w_bench from total market-cap of ALL
   stocks in meta_df on that day (not just the tradable subset), so the
   benchmark reflects the true index composition.
4. Build the industry dummy matrix X_ind for S_t.
5. Extract w_prev for S_t from the full previous-day weight vector (new
   entrants receive 0; forced exits are zeroed out in the full vector,
   generating real turnover).
6. De-mean alpha_t cross-sectionally.
7. Call optimizer.solve(alpha_centered_t, w_prev, X_ind, w_bench) → w_t*.
8. Reconstruct the full N-stock weight vector: S_t stocks get w_t*, all
   others get 0.

Performance metrics
--------------------
Gross return, transaction costs (at cost_rate), net return, annualised return
/ vol, Sharpe ratio, max drawdown, average daily turnover, breakeven turnover.
Identical to net_backtester.py's metric set for easy comparison.

A summary of industry-tolerance relaxation events is also written to the
report, showing how often and how much the industry constraint was relaxed.

Public API
----------
    obt = OptimizationBacktester(alpha_df, prices_df, meta_df,
                                  cost_rate=0.002,
                                  lambda_turnover=0.2,
                                  rf=0.03, max_weight=0.05,
                                  industry_tol=0.01, plots_dir=None)
    summary = obt.run_backtest()   # pd.Series of performance metrics
    fig     = obt.plot(show=False) # cumulative NAV chart
"""

from __future__ import annotations

import math
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from optimizer import PortfolioOptimizer


class OptimizationBacktester:
    """Day-by-day optimised portfolio backtester.

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Flat [trade_date, ts_code, ml_alpha].  Exactly one non-key column.
    prices_df : pd.DataFrame
        Flat [trade_date, ts_code, close, tradable, ...].
    meta_df : pd.DataFrame
        Flat [trade_date, ts_code, industry, total_mv, ...].
    cost_rate : float
        Actual one-way transaction cost rate used for P&L accounting
        (default 0.002 = 0.2 %).  Applied as:
            net_return = gross_return - turnover * cost_rate.
        This is a *monetary rate*, NOT the optimiser penalty coefficient.
        See ``lambda_turnover`` for the optimiser-side parameter.
    lambda_turnover : float
        Turnover-aversion coefficient in the optimiser objective (default 0.2).
        Controls the signal-vs-stability trade-off inside the LP.  This is a
        *dimensionless policy parameter*, NOT a monetary cost rate.
        Tune by inspecting ``Avg Daily Turnover`` in the backtest report:
          - 0.05~0.1  : high turnover, ~10–20 % daily
          - 0.2~0.5   : moderate, ~2–8 % daily  ← recommended starting point
          - 1.0+      : very stable, slow signal tracking
    rf : float
        Annual risk-free rate for Sharpe ratio (default 0.03).
    max_weight : float
        Per-stock weight cap passed to the optimiser (default 0.05 = 5%).
    industry_tol : float
        Initial industry-deviation tolerance passed to the optimiser
        (default 0.01 = ±1 pp).  Auto-relaxed up to 5% on infeasible days.
    plots_dir : path-like, optional
        Directory to save the NAV chart.  None disables file output.
    """

    TRADING_DAYS_PER_YEAR: int = 252

    def __init__(
        self,
        alpha_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        cost_rate: float = 0.002,
        lambda_turnover: float = 0.2,
        rf: float = 0.03,
        max_weight: float = 0.05,
        industry_tol: float = 0.01,
        plots_dir: Optional[pathlib.Path | str] = None,
    ) -> None:
        # Identify alpha column
        key_cols = {"trade_date", "ts_code"}
        alpha_cols = [c for c in alpha_df.columns if c not in key_cols]
        if len(alpha_cols) != 1:
            raise ValueError(
                f"alpha_df must have exactly one alpha column; got {alpha_cols}"
            )
        self.alpha_col = alpha_cols[0]
        self.cost_rate = cost_rate           # monetary rate for P&L accounting
        self.lambda_turnover = lambda_turnover  # optimiser penalty coefficient
        self.rf = rf
        self.max_weight = max_weight
        self.industry_tol = industry_tol
        self.plots_dir = pathlib.Path(plots_dir) if plots_dir is not None else None

        self._alpha_df  = alpha_df
        self._prices_df = prices_df
        self._meta_df   = meta_df

        # Cached results
        self._net_ret: Optional[pd.Series]   = None
        self._gross_ret: Optional[pd.Series] = None
        self._turnover: Optional[pd.Series]  = None
        self._summary: Optional[pd.Series]   = None
        self._relax_log: list[dict]           = []   # records of tolerance relaxation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _perf_metrics(r: pd.Series, rf: float, tdy: int) -> dict:
        """Standard performance metrics from a daily-return series."""
        r = r.dropna()
        n = len(r)
        if n == 0:
            return {k: float("nan") for k in
                    ["Cum Return", "Ann Return", "Ann Vol", "Sharpe", "Max DD"]}
        cum_val = (1 + r).cumprod()
        cum_ret = cum_val.iloc[-1] - 1
        ann_ret = (1 + cum_ret) ** (tdy / n) - 1
        ann_vol = r.std() * math.sqrt(tdy)
        sharpe  = (ann_ret - rf) / ann_vol if ann_vol != 0 else float("nan")
        mdd     = (cum_val / cum_val.cummax() - 1).min()
        return {
            "Cum Return": cum_ret,
            "Ann Return": ann_ret,
            "Ann Vol":    ann_vol,
            "Sharpe":     sharpe,
            "Max DD":     mdd,
        }

    def _build_return_series(self) -> None:
        """Core loop: call the optimiser for each day and accumulate returns.

        Execution assumption (open-to-open, no look-ahead bias)
        ---------------------------------------------------------
        The portfolio decision for day T is made from the alpha signal computed
        at the close of day T.  The trade executes at the OPEN of day T+1 and
        is held until the OPEN of day T+2.  Therefore, the realised stock return
        each period is the consecutive open-to-open return:

            stock_ret[T] = open_T / open_{T-1} - 1

        This aligns exactly with ``calc_forward_return`` in targets.py:
            forward_return_T = open_{T+2} / open_{T+1} - 1
        ensuring that what the LP maximises (expected forward return) matches
        what the backtest actually realises.
        """
        if self._net_ret is not None:
            return

        # ---- Prepare wide price and tradability tables ----------------
        open_wide = self._prices_df.pivot(
            index="trade_date", columns="ts_code", values="open"
        ).sort_index()
        tradable_wide = self._prices_df.pivot(
            index="trade_date", columns="ts_code", values="tradable"
        ).sort_index().fillna(False).astype(bool)

        # Open-to-open daily stock returns
        stock_ret_wide = open_wide.pct_change()

        # ---- Prepare alpha wide table --------------------------------
        alpha_wide = self._alpha_df.pivot(
            index="trade_date", columns="ts_code", values=self.alpha_col
        ).sort_index()

        # ---- Align to a common date × stock universe ------------------
        all_dates  = alpha_wide.index.intersection(open_wide.index)
        all_stocks = (
            alpha_wide.columns
            .union(open_wide.columns)
            .union(tradable_wide.columns)
        )
        alpha_wide    = alpha_wide.reindex(index=all_dates, columns=all_stocks)
        open_wide     = open_wide.reindex(index=all_dates, columns=all_stocks)
        tradable_wide = tradable_wide.reindex(
            index=all_dates, columns=all_stocks, fill_value=False
        )
        stock_ret_wide = open_wide.pct_change()   # recompute after reindex

        # Prepare meta (for industry labels and market cap)
        meta_cols = ["trade_date", "ts_code", "industry", "total_mv"]
        available_meta = [c for c in meta_cols if c in self._meta_df.columns]
        meta_sub = self._meta_df[available_meta].copy()
        has_mv = "total_mv" in meta_sub.columns

        # ---- Initialise full portfolio weight vector ------------------
        w_full = pd.Series(0.0, index=all_stocks)

        # ---- Pre-index meta by date for speed ------------------------
        meta_by_date = {
            date: grp.set_index("ts_code")
            for date, grp in meta_sub.groupby("trade_date")
        }

        # ---- Main loop -----------------------------------------------
        # lambda_turnover: optimiser signal-vs-stability knob (NOT a cost rate)
        # cost_rate: used only for net-return accounting below
        optimiser = PortfolioOptimizer(
            lambda_turnover=self.lambda_turnover,
            max_weight=self.max_weight,
            industry_tol=self.industry_tol,
        )

        gross_ret_list: list[float] = []
        turnover_list:  list[float] = []
        net_ret_list:   list[float] = []
        date_index:     list        = []

        self._relax_log = []

        for date in all_dates:
            # Gross return using previous-day weights
            ret_today = stock_ret_wide.loc[date].fillna(0.0)
            gross = float((w_full * ret_today).sum())

            # Determine tradable universe for today
            tradable_row = tradable_wide.loc[date]
            alpha_row    = alpha_wide.loc[date]
            valid_mask   = tradable_row & alpha_row.notna()
            S_t = all_stocks[valid_mask].tolist()

            if len(S_t) >= 2:
                # Extract raw alpha and previous weights for S_t
                alpha_t = alpha_row[S_t].values.astype(float)
                w_prev  = w_full[S_t].values.astype(float)

                # Cross-sectional de-mean: centres signal around 0 so that
                # lambda_turnover operates in a stable numeric range.
                # This does NOT change the cross-sectional rank ordering.
                alpha_t = alpha_t - np.nanmean(alpha_t)

                # Build benchmark industry weights from ALL stocks on this day
                meta_today = meta_by_date.get(date)
                industries, X_ind, w_bench = self._build_industry_inputs(
                    S_t, meta_today, has_mv
                )

                # Solve the LP
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    w_star, tol_used = optimiser.solve(
                        alpha_t, w_prev, X_ind, w_bench
                    )
                    for w in caught:
                        self._relax_log.append({
                            "date": date,
                            "message": str(w.message),
                        })

                # Update full weight vector
                w_new = pd.Series(0.0, index=all_stocks)
                w_new[S_t] = w_star
                # Log if industry tolerance was relaxed beyond initial value
                if tol_used is not None and tol_used > self.industry_tol:
                    self._relax_log.append({
                        "date": date,
                        "message": f"industry tol relaxed to {tol_used:.2%}",
                    })
            else:
                # Too few tradable stocks: hold current position (no trade)
                w_new = w_full.copy()
                w_new[~w_full.index.isin(S_t)] = 0.0

            # Turnover = half L1 norm of weight change (one-way)
            turnover = 0.5 * float((w_new - w_full).abs().sum())

            # Net return: deduct actual trading costs (uses cost_rate, not lambda)
            net = gross - turnover * self.cost_rate

            gross_ret_list.append(gross)
            turnover_list.append(turnover)
            net_ret_list.append(net)
            date_index.append(date)

            w_full = w_new

        # Skip the very first row (no previous-day weights, gross=0)
        self._gross_ret = pd.Series(gross_ret_list[1:], index=date_index[1:])
        self._turnover  = pd.Series(turnover_list[1:],  index=date_index[1:])
        self._net_ret   = pd.Series(net_ret_list[1:],   index=date_index[1:])

    def _build_industry_inputs(
        self,
        stocks: list[str],
        meta_today: Optional[pd.DataFrame],
        has_mv: bool,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Build X_industry and w_benchmark for the given stock list.

        Benchmark weights are computed from total_mv across ALL stocks present
        in meta_today (including non-tradable ones) to reflect the true index
        composition.  Falls back to equal-weight per industry when total_mv is
        unavailable.

        Returns
        -------
        industries : list[str]
            Sorted list of K industry names.
        X_ind : np.ndarray, shape (n, K)
            Binary industry dummy matrix for ``stocks``.
        w_bench : np.ndarray, shape (K,)
            Benchmark industry weight vector (sums to 1).
        """
        if meta_today is None or meta_today.empty:
            # Fallback: all stocks in one dummy industry, equal weight
            n = len(stocks)
            return ["ALL"], np.ones((n, 1)), np.array([1.0])

        # Industry label for each stock in S_t (tradable universe)
        stock_industries = meta_today["industry"].reindex(stocks).fillna("OTHER")
        industries = sorted(stock_industries.unique().tolist())
        K = len(industries)
        n = len(stocks)

        # X_ind: shape (n, K)
        ind_to_idx = {ind: i for i, ind in enumerate(industries)}
        X_ind = np.zeros((n, K), dtype=float)
        for row_i, stock in enumerate(stocks):
            ind = stock_industries.get(stock, "OTHER")
            col_j = ind_to_idx.get(ind, ind_to_idx.get("OTHER", 0))
            X_ind[row_i, col_j] = 1.0

        # Benchmark weights from ALL stocks' market cap
        if has_mv and "total_mv" in meta_today.columns:
            all_mv   = meta_today["total_mv"].fillna(0.0)
            all_inds = meta_today["industry"].fillna("OTHER")
            mv_total = all_mv.sum()
            if mv_total > 0:
                ind_mv = (
                    pd.Series(all_mv.values, index=all_inds.values)
                    .groupby(level=0)
                    .sum()
                    .reindex(industries, fill_value=0.0)
                )
                w_bench = (ind_mv / mv_total).values.astype(float)
            else:
                # No valid market cap data: equal weight
                w_bench = np.full(K, 1.0 / K)
        else:
            # total_mv not available: equal weight across industries
            w_bench = np.full(K, 1.0 / K)

        # Normalise (guard against floating-point drift)
        bench_sum = w_bench.sum()
        if bench_sum > 0:
            w_bench = w_bench / bench_sum

        return industries, X_ind, w_bench

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(self) -> pd.Series:
        """Execute the optimised backtest.

        Returns
        -------
        pd.Series
            Performance metrics: Cum Return, Ann Return, Ann Vol, Sharpe,
            Max DD, Avg Daily Turnover, Breakeven Turnover,
            Relaxation Events (count of days the industry tol was widened).
        """
        self._build_return_series()

        tdy = self.TRADING_DAYS_PER_YEAR
        metrics = self._perf_metrics(self._net_ret, self.rf, tdy)

        gross_metrics = self._perf_metrics(self._gross_ret, self.rf, tdy)
        ann_gross_ret = gross_metrics["Ann Return"]

        avg_turnover = float(self._turnover.mean())
        breakeven = (
            (ann_gross_ret - self.rf) / (self.cost_rate * tdy)
            if self.cost_rate > 0 else float("nan")
        )

        metrics["Avg Daily Turnover"]  = avg_turnover
        metrics["Breakeven Turnover"]  = breakeven
        metrics["Relaxation Events"]   = sum(
            1 for e in self._relax_log if "relaxed" in e.get("message", "")
        )

        self._summary = pd.Series(metrics).round(6)
        return self._summary

    def plot(self, show: bool = False) -> plt.Figure:
        """Plot the cumulative net return NAV curve.

        Saves to ``plots_dir/optimization_nav.png`` when ``plots_dir`` is set.
        """
        self._build_return_series()

        cum_nav = (1 + self._net_ret).cumprod()

        try:
            dates = pd.to_datetime(cum_nav.index.astype(str), format="%Y%m%d")
        except Exception:
            dates = pd.to_datetime(cum_nav.index, errors="coerce")

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(
            f"Optimised Portfolio — Net Return (after costs)\n"
            f"lambda_TO={self.lambda_turnover},  cost={self.cost_rate:.2%},  "
            f"max_w={self.max_weight:.0%},  ind_tol=±{self.industry_tol:.0%}",
            fontsize=11,
            fontweight="bold",
        )

        ax.plot(dates, cum_nav.values, color="#1565C0", linewidth=1.8,
                label="Optimised Net NAV")
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Cumulative NAV")
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        if self._summary is not None:
            txt = (
                f"Ann Ret : {self._summary['Ann Return']:+.2%}\n"
                f"Sharpe  : {self._summary['Sharpe']:.3f}\n"
                f"Max DD  : {self._summary['Max DD']:.2%}\n"
                f"Avg TO  : {self._summary['Avg Daily Turnover']:.4f}"
            )
            ax.text(
                0.88, 0.05, txt,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", alpha=0.15),
            )

        locator   = mdates.AutoDateLocator(minticks=6, maxticks=12)
        formatter = mdates.DateFormatter("%Y-%m")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        if self.plots_dir is not None:
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.plots_dir / "optimization_nav.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return fig

    def print_relax_summary(self) -> None:
        """Print a brief summary of industry-tolerance relaxation events."""
        events = [e for e in self._relax_log if "relaxed" in e.get("message", "")]
        if not events:
            print("No industry-tolerance relaxation events.")
            return
        print(f"\nIndustry tolerance relaxation events ({len(events)} days):")
        for e in events[:20]:
            print(f"  {e['date']} — {e['message']}")
        if len(events) > 20:
            print(f"  ... and {len(events) - 20} more.")
