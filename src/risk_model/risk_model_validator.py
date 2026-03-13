"""
risk_model_validator.py
-----------------------
Validate the multi-factor risk model by comparing predicted portfolio variance
against realized variance of a CSI 300 market-cap weighted portfolio.

Compares:
  - Predicted variance: w' Sigma w from risk model (factor + idiosyncratic)
  - Realized variance: rolling sample variance of portfolio returns

Usage
-----
    Called from risk_model_main.py after saving risk_*.parquet.
    Not intended for standalone execution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy import stats


class RiskModelValidator:
    """Validate risk model by comparing predicted vs realized portfolio variance.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Flat [trade_date, ts_code, close, tradable, ...].
    meta_df : pd.DataFrame
        Flat [trade_date, ts_code, total_mv, ...].
    exposure_df : pd.DataFrame
        Long format [trade_date, ts_code, <factor_cols>] from risk_exposure.
    f_half_df : pd.DataFrame
        Long format [trade_date, f_i, f_j, value] from risk_cov_F (Cholesky L^T).
    delta_df : pd.DataFrame
        Long format [trade_date, ts_code, delta_std] from risk_delta.
    realized_window : int
        Rolling window for realized variance (default 20 days).
    min_stocks : int
        Minimum stocks required per day to include in validation (default 10).
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        exposure_df: pd.DataFrame,
        f_half_df: pd.DataFrame,
        delta_df: pd.DataFrame,
        realized_window: int = 60,
        min_stocks: int = 10,
    ) -> None:
        self.realized_window = realized_window
        self.min_stocks = min_stocks

        # Pivot to wide format for daily operations
        self._close_wide = (
            prices_df.pivot(index="trade_date", columns="ts_code", values="close")
            .sort_index()
        )
        self._tradable_wide = (
            prices_df.pivot(index="trade_date", columns="ts_code", values="tradable")
            .sort_index()
            .fillna(False)
            .astype(bool)
        )
        self._mv_wide = (
            meta_df.pivot(index="trade_date", columns="ts_code", values="total_mv")
            .sort_index()
            .reindex(self._close_wide.index)
        )
        self._returns_wide = self._close_wide.pct_change()

        self._all_dates = self._returns_wide.index
        self._all_stocks = self._returns_wide.columns.tolist()

        # Index risk model by date
        exposure_df = exposure_df.copy()
        exposure_df["trade_date"] = pd.to_datetime(exposure_df["trade_date"])
        factor_cols = [c for c in exposure_df.columns if c not in {"trade_date", "ts_code"}]
        self._factor_cols = factor_cols
        self._exposure_by_date = {
            date: grp.set_index("ts_code")[factor_cols]
            for date, grp in exposure_df.groupby("trade_date")
        }

        f_half_df = f_half_df.copy()
        f_half_df["trade_date"] = pd.to_datetime(f_half_df["trade_date"])
        self._f_half_by_date = {}
        for date, grp in f_half_df.groupby("trade_date"):
            pivot = grp.pivot(index="f_i", columns="f_j", values="value")
            pivot = pivot.reindex(index=factor_cols, columns=factor_cols, fill_value=0.0)
            self._f_half_by_date[date] = pivot.values.astype(float)

        delta_df = delta_df.copy()
        delta_df["trade_date"] = pd.to_datetime(delta_df["trade_date"])
        self._delta_by_date = {
            date: grp.set_index("ts_code")["delta_std"]
            for date, grp in delta_df.groupby("trade_date")
        }

    def run_validation(
        self,
    ) -> tuple[pd.Series, pd.Series, dict]:
        """Run validation: predicted vs realized variance.

        Returns
        -------
        predicted_var : pd.Series
            Daily predicted variance (index=trade_date).
        realized_var : pd.Series
            Rolling realized variance (index=trade_date).
        metrics : dict
            R², Pearson corr, Spearman corr, bias ratio, RMSE (variance), RMSE (vol).
        """
        # 1. Build daily market-cap weights and portfolio returns
        port_ret_list = []
        date_list = []
        w_prev = None

        for i, date in enumerate(self._all_dates):
            mv_row = self._mv_wide.loc[date]
            tradable_row = self._tradable_wide.loc[date]
            ret_row = self._returns_wide.loc[date]

            mask = (
                tradable_row.fillna(False)
                & mv_row.notna()
                & (mv_row > 0)
            )
            stocks_t = [s for s in self._all_stocks if mask.get(s, False)]

            if len(stocks_t) < self.min_stocks:
                continue

            mv_t = mv_row[stocks_t].values.astype(float)
            w_t = mv_t / mv_t.sum()

            if w_prev is not None:
                # Portfolio return: r_t = w_{t-1}' * ret_t (full universe)
                ret_full = ret_row.reindex(self._all_stocks).fillna(0.0)
                r_port = float((w_prev * ret_full).sum())
                port_ret_list.append(r_port)
                date_list.append(date)

            w_prev = pd.Series(0.0, index=self._all_stocks)
            w_prev[stocks_t] = w_t

        port_ret = pd.Series(port_ret_list, index=date_list)

        # 2. Rolling realized variance
        realized_var = port_ret.rolling(
            self.realized_window, min_periods=self.realized_window
        ).var()

        # 3. Predicted variance per day
        pred_list = []
        pred_dates = []

        for date in realized_var.dropna().index:
            date_norm = pd.to_datetime(date)
            exp_day = self._exposure_by_date.get(date_norm)
            f_half = self._f_half_by_date.get(date_norm)
            delta_day = self._delta_by_date.get(date_norm)

            if exp_day is None or f_half is None or delta_day is None:
                continue

            # Build weights for this date
            mv_row = self._mv_wide.loc[date]
            tradable_row = self._tradable_wide.loc[date]
            mask = tradable_row & mv_row.notna() & (mv_row > 0)
            stocks_t = [s for s in self._all_stocks if mask.get(s, False)]

            if len(stocks_t) < self.min_stocks:
                continue

            mv_t = mv_row[stocks_t].values.astype(float)
            w = (mv_t / mv_t.sum()).astype(float)

            # Align to risk model
            X_sub = exp_day.reindex(stocks_t).fillna(0.0)
            delta_sub = delta_day.reindex(stocks_t)
            med = float(delta_day.median()) if not delta_day.empty else 1e-3
            delta_sub = delta_sub.fillna(med)

            X_arr = X_sub.values.astype(float)
            delta_arr = delta_sub.values.astype(float)

            z = f_half @ (X_arr.T @ w)
            day_var = float(np.dot(z, z) + np.dot(delta_arr * w, delta_arr * w))

            pred_list.append(day_var)
            pred_dates.append(date)

        predicted_var = pd.Series(pred_list, index=pred_dates)

        # 4. Align and compute metrics
        common = predicted_var.index.intersection(realized_var.dropna().index)
        pred_aligned = predicted_var.reindex(common).dropna()
        real_aligned = realized_var.reindex(common).dropna()
        common = pred_aligned.index.intersection(real_aligned.index)
        pred_aligned = pred_aligned.loc[common]
        real_aligned = real_aligned.loc[common]

        if len(common) < 2:
            metrics = {
                "R2": float("nan"),
                "Pearson_corr": float("nan"),
                "Spearman_corr": float("nan"),
                "Bias_ratio": float("nan"),
                "RMSE_var": float("nan"),
                "RMSE_vol": float("nan"),
                "N_days": len(common),
            }
            return predicted_var, realized_var, metrics

        # R² from linear regression realized ~ predicted
        _, _, r_val, _, _ = stats.linregress(pred_aligned.values, real_aligned.values)
        r2 = r_val ** 2

        pearson = float(np.corrcoef(pred_aligned.values, real_aligned.values)[0, 1])
        spearman = float(stats.spearmanr(pred_aligned.values, real_aligned.values)[0])

        bias_ratio = float(pred_aligned.mean() / real_aligned.mean()) if real_aligned.mean() != 0 else float("nan")

        rmse_var = float(np.sqrt(((pred_aligned - real_aligned) ** 2).mean()))
        rmse_vol = float(np.sqrt(((np.sqrt(pred_aligned) - np.sqrt(real_aligned)) ** 2).mean()))

        metrics = {
            "R2": round(r2, 6),
            "Pearson_corr": round(pearson, 6),
            "Spearman_corr": round(spearman, 6),
            "Bias_ratio": round(bias_ratio, 6),
            "RMSE_var": round(rmse_var, 8),
            "RMSE_vol": round(rmse_vol, 6),
            "N_days": len(common),
        }

        return predicted_var, realized_var, metrics

    def plot(
        self,
        predicted_var: pd.Series,
        realized_var: pd.Series,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """Plot predicted vs realized variance (time series and scatter).

        Parameters
        ----------
        predicted_var : pd.Series
            From run_validation().
        realized_var : pd.Series
            From run_validation().
        save_path : str or None
            Path to save figure.
        show : bool
            Whether to display the plot.
        """
        common = predicted_var.index.intersection(realized_var.dropna().index)
        pred = predicted_var.reindex(common).dropna()
        real = realized_var.reindex(common).dropna()
        common = pred.index.intersection(real.index)
        pred = pred.loc[common]
        real = real.loc[common]

        if len(common) < 2:
            return

        # Convert index to datetime for correct matplotlib date axis
        # (trade_date may be int YYYYMMDD from parquet; raw int would show as 1970)
        if pd.api.types.is_datetime64_any_dtype(common):
            common_dt = common
        else:
            common_dt = pd.to_datetime(common.astype(str), format="%Y%m%d", errors="coerce")
            if common_dt.isna().any():
                common_dt = pd.to_datetime(common)

        # Side-by-side layout: (a) time series left (wider), (b) scatter right
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor("white")
        for ax in axes:
            ax.set_facecolor("white")

        fig.suptitle(
            "Risk Model Validation: Predicted vs Realized Volatility",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )

        # Left: time series (volatility = sqrt(variance) for readability)
        ax0 = axes[0]
        ax0.plot(
            common_dt,
            np.sqrt(pred.values) * 100,
            label="Predicted vol (%)",
            color="#1565C0",
            linewidth=1.5,
            alpha=0.9,
        )
        ax0.plot(
            common_dt,
            np.sqrt(real.values) * 100,
            label="Realized vol (%)",
            color="#E65100",
            linewidth=1.5,
            alpha=0.9,
        )
        ax0.set_title("(a) Daily volatility time series", fontsize=10, loc="left")
        ax0.set_ylabel("Daily volatility (%)")
        ax0.legend(loc="upper right", fontsize=9)
        ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax0.grid(True, alpha=0.25, linestyle="-", color="gray")
        locator0 = mdates.AutoDateLocator(minticks=6, maxticks=12)
        ax0.xaxis.set_major_locator(locator0)
        ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax0.tick_params(axis="x", rotation=30)

        # Right: scatter (variance space) — square aspect
        ax1 = axes[1]
        ax1.scatter(real.values, pred.values, alpha=0.5, s=14, color="#1565C0", edgecolors="none")
        max_val = max(real.max(), pred.max())
        ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.6, linewidth=1.2, label="y = x")
        ax1.set_title("(b) Predicted vs realized variance", fontsize=10, loc="left")
        ax1.set_xlabel("Realized variance")
        ax1.set_ylabel("Predicted variance")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.25, linestyle="-", color="gray")
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlim(0, max_val * 1.02)
        ax1.set_ylim(0, max_val * 1.02)

        fig.autofmt_xdate(rotation=30)
        fig.tight_layout(rect=[0, 0.06, 1, 0.98])
        fig.text(
            0.5,
            0.02,
            "CSI 300 market-cap weighted portfolio. "
            "Predicted: w'Σw from risk model; Realized: rolling variance of portfolio returns.",
            ha="center",
            fontsize=8,
            color="gray",
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)
