"""
FactorEngine: compute all feature factors for the risk model.

Factor groups
-------------
A — Microstructure and price-volume factors (use adjusted prices):
    amihud_illiq      : Amihud (2002) illiquidity ratio, 20-day mean
    ivol              : idiosyncratic volatility vs CSI 300, rolling 20-day OLS residual std
    realized_skewness : rolling 20-day skewness of daily returns
    vol_price_corr    : rolling 10-day Spearman rank correlation between price and volume

B — Fundamental and valuation factors (point-in-time quarterly data):
    ep  : earnings-to-price  (1 / PE)
    bp  : book-to-price      (1 / PB)
    roe : return on equity, quarterly, PIT-aligned via ann_date

C — Cross-sectional relative features:
    industry_rel_turnover : turnover_rate − industry cross-section median
    industry_rel_bp       : BP − industry cross-section median
    ts_rel_turnover       : today's turnover_rate / 60-day rolling mean

Price adjustment (adj_type parameter):
    'forward'  (default) : P × adj_factor / adj_factor_latest  (last row in df_adj)
    'backward'           : P × adj_factor
    'raw'                : no adjustment

PIT alignment note:
    For B-group factors, quarterly data is forward-filled from the announcement date
    (ann_date).  merge_asof with direction='backward' ensures that for any trading
    date t, only announcements with ann_date ≤ t are used.  No future information
    leaks into the factor values.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class FactorEngine:
    """
    Compute all feature factors from the project data dictionary.

    Parameters
    ----------
    data_dict : dict
        Returned by DataEngine.load_data().  Must contain:
            df_price, df_mv, df_basic, df_industry, df_adj,
            df_index, df_financials
    adj_type : str, default 'forward'
        Price adjustment mode: 'forward' | 'backward' | 'raw'
    latest_adj : pd.Series or None
        Optional pre-fetched latest adj_factor values (for forward adjustment).
        When None, the last row of df_adj is used.
    """

    def __init__(
        self,
        data_dict: dict,
        adj_type: str = "forward",
        latest_adj: pd.Series = None,
    ):
        if adj_type not in ("forward", "backward", "raw"):
            raise ValueError(f"adj_type must be 'forward', 'backward', or 'raw'; got '{adj_type}'")

        df_price = data_dict["df_price"]
        df_mv    = data_dict["df_mv"]
        df_adj   = data_dict.get("df_adj")
        df_basic = data_dict["df_basic"]

        # ---- Unstack raw price series (wide form: dates × codes) ----
        open_raw  = df_price["open"].unstack("code")
        high_raw  = df_price["high"].unstack("code")
        low_raw   = df_price["low"].unstack("code")
        close_raw = df_price["close"].unstack("code")
        self.vol  = df_price["vol"].unstack("code")     # never adjusted

        # ---- Precise VWAP: amount (thousand CNY) × 10 / vol (lots) ----
        # amount (千元) × 1000 / (vol (手) × 100) = amount × 10 / vol  (元/股)
        if "amount" in df_price.columns:
            amount_wide = df_price["amount"].unstack("code")
            vwap_raw = (amount_wide * 10).div(self.vol)
            # Fallback to typical price where amount or vol is missing / zero
            fallback = (high_raw + low_raw + close_raw) / 3
            valid = amount_wide.notna() & (self.vol > 0)
            vwap_raw = vwap_raw.where(valid, fallback)
        else:
            vwap_raw = (high_raw + low_raw + close_raw) / 3

        # Raw amount (千元 → 元 conversion applied inside amihud_illiq)
        if "amount" in df_price.columns:
            self.amount_raw = df_price["amount"].unstack("code")  # unit: 千元
        else:
            self.amount_raw = None

        # ---- Price adjustment ----
        if adj_type != "raw" and df_adj is not None and not df_adj.empty:
            adj_wide = df_adj["adj_factor"].unstack("code").reindex(open_raw.index).ffill()
            if adj_type == "forward":
                latest_row = adj_wide.iloc[-1] if latest_adj is None else latest_adj.reindex(adj_wide.columns)
                adj_ratio = adj_wide.div(latest_row, axis=1)
            else:
                adj_ratio = adj_wide
            self.open  = open_raw.mul(adj_ratio)
            self.high  = high_raw.mul(adj_ratio)
            self.low   = low_raw.mul(adj_ratio)
            self.close = close_raw.mul(adj_ratio)
            self.vwap = vwap_raw.mul(adj_ratio)
        else:
            self.open  = open_raw
            self.high  = high_raw
            self.low   = low_raw
            self.close = close_raw
            self.vwap = vwap_raw

        self.returns  = self.close.pct_change()
        self.total_mv = df_mv["total_mv"].unstack("code")

        # ---- Daily basic fields ----
        self.pe            = df_basic["pe"].unstack("code") if "pe" in df_basic.columns else None
        self.pb            = df_basic["pb"].unstack("code") if "pb" in df_basic.columns else None
        self.turnover_rate = df_basic["turnover_rate"].unstack("code") if "turnover_rate" in df_basic.columns else None
        self.volume_ratio  = df_basic["volume_ratio"].unstack("code") if "volume_ratio" in df_basic.columns else None
        self.ps_ttm       = df_basic["ps_ttm"].unstack("code") if "ps_ttm" in df_basic.columns else None
        self.dv_ratio     = df_basic["dv_ratio"].unstack("code") if "dv_ratio" in df_basic.columns else None
        self.circ_mv      = df_basic["circ_mv"].unstack("code") if "circ_mv" in df_basic.columns else None

        # ---- Industry mapping (code → SW L1 name) ----
        df_industry = data_dict["df_industry"]
        self._industry = df_industry["industry"].reindex(self.close.columns)

        # ---- Index returns (for IVOL) ----
        index_close = data_dict.get("df_index")
        if index_close is not None and not index_close.empty:
            index_close = index_close.reindex(self.close.index).ffill()
            self.market_ret = index_close.pct_change()
        else:
            self.market_ret = pd.Series(dtype=float)

        # ---- Quarterly financials (raw, for PIT alignment) ----
        self._df_financials = data_dict.get("df_financials", pd.DataFrame())

    # ==================================================================
    # Utility functions (time-series and cross-sectional operators)
    # ==================================================================

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank, normalized to (0, 1]."""
        return df.rank(axis=1, pct=True)

    def _delay(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Value d days ago."""
        return df.shift(d)

    def _delta(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Change over d days."""
        return df.diff(d)

    def _corr(self, df1: pd.DataFrame, df2: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day time-series Pearson correlation, per stock."""
        return df1.rolling(d).corr(df2)

    def _cov(self, df1: pd.DataFrame, df2: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day covariance, per stock."""
        return df1.rolling(d).cov(df2)

    def _stddev(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day standard deviation, per stock."""
        return df.rolling(d).std()

    def _sum(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day sum, per stock."""
        return df.rolling(d).sum()

    def _product(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day product, per stock."""
        return df.rolling(d).apply(np.prod, raw=True)

    def _ts_min(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day minimum, per stock."""
        return df.rolling(d).min()

    def _ts_max(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day maximum, per stock."""
        return df.rolling(d).max()

    def _ts_argmax(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """0-based offset of rolling maximum over d days."""
        return df.rolling(d).apply(np.argmax, raw=True)

    def _ts_argmin(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """0-based offset of rolling minimum over d days."""
        return df.rolling(d).apply(np.argmin, raw=True)

    def _ts_rank(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """
        Time-series rank of today's value within the past d days.
        Normalized to (0, 1].
        """
        def _rank_last(arr: np.ndarray) -> float:
            temp = arr.argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(1, len(arr) + 1)
            return ranks[-1] / len(arr)
        return df.rolling(d).apply(_rank_last, raw=True)

    def _scale(self, df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        """Cross-sectional rescaling so sum(abs(x)) = a."""
        abs_sum = df.abs().sum(axis=1).replace(0, np.nan)
        return df.div(abs_sum, axis=0) * a

    def _decay_linear(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Linearly-decaying weighted average over d days (most recent = weight d)."""
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()
        return df.rolling(d).apply(lambda x: np.dot(x, weights), raw=True)

    def _sign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Element-wise sign."""
        return np.sign(df)

    def _log(self, df: pd.DataFrame) -> pd.DataFrame:
        """Element-wise natural logarithm."""
        return np.log(df)

    def _abs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Element-wise absolute value."""
        return df.abs()

    def _signed_power(self, df: pd.DataFrame, e: float) -> pd.DataFrame:
        """sign(x) * |x|^e."""
        return np.sign(df) * (np.abs(df) ** e)

    def _skew(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day skewness per stock, using scipy for bias correction."""
        return df.rolling(d).apply(
            lambda x: scipy_stats.skew(x[~np.isnan(x)], bias=False) if len(x[~np.isnan(x)]) >= 3 else np.nan,
            raw=True,
        )

    def _kurt(self, df: pd.DataFrame, d: int) -> pd.DataFrame:
        """Rolling d-day excess kurtosis per stock (bias-corrected, Fisher definition).
        Requires at least 4 valid observations in the window.
        """
        return df.rolling(d).apply(
            lambda x: scipy_stats.kurtosis(x[~np.isnan(x)], bias=False, fisher=True)
            if len(x[~np.isnan(x)]) >= 4 else np.nan,
            raw=True,
        )

    def _rolling_ols(self, d: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Vectorized rolling OLS: r_stock = alpha + beta * r_market + epsilon.

        For each trading date t, fits using the previous d days.
        Returns (beta_df, ivol_df) where:
            beta_df: market beta coefficient (slope)
            ivol_df: std(epsilon), idiosyncratic volatility

        Vectorized across stocks; stocks with NaN in window receive NaN.
        """
        mkt = self.market_ret.reindex(self.returns.index).values
        ret_arr = self.returns.values
        T, N = ret_arr.shape
        beta_arr = np.full((T, N), np.nan)
        ivol_arr = np.full((T, N), np.nan)

        for t in range(d - 1, T):
            w_mkt = mkt[t - d + 1: t + 1]
            if np.any(np.isnan(w_mkt)):
                continue

            w_ret = ret_arr[t - d + 1: t + 1, :]
            X = np.column_stack([np.ones(d), w_mkt])

            try:
                XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
            except np.linalg.LinAlgError:
                continue

            valid_stocks = ~np.any(np.isnan(w_ret), axis=0)
            if not valid_stocks.any():
                continue

            Y = w_ret[:, valid_stocks]
            coef = XtX_inv_Xt @ Y  # shape (2, n_valid): [alpha, beta]
            resid = Y - X @ coef
            beta_arr[t, valid_stocks] = coef[1, :]  # market beta
            ivol_arr[t, valid_stocks] = np.std(resid, axis=0, ddof=2)

        idx = self.returns.index
        cols = self.returns.columns
        return (
            pd.DataFrame(beta_arr, index=idx, columns=cols),
            pd.DataFrame(ivol_arr, index=idx, columns=cols),
        )

    def _rolling_ols_resid_std(self, d: int) -> pd.DataFrame:
        """IVOL from rolling OLS; delegates to _rolling_ols."""
        _, ivol_df = self._rolling_ols(d)
        return ivol_df

    def _pit_align_financial(self, col: str) -> pd.DataFrame:
        """
        Forward-fill a quarterly financial series to daily frequency using PIT.

        For each trading date t and each stock, the value returned is the most
        recently announced value with ann_date <= t (merge_asof, backward fill).

        Returns wide DataFrame: index = trade date, columns = stock code.
        """
        if self._df_financials.empty or col not in self._df_financials.columns:
            return pd.DataFrame(
                np.nan, index=self.close.index, columns=self.close.columns
            )

        fin = (
            self._df_financials[["code", "ann_date", col]]
            .dropna(subset=[col])
            .sort_values("ann_date")
        )

        trade_dates = self.close.index
        codes = self.close.columns
        result = pd.DataFrame(np.nan, index=trade_dates, columns=codes)
        # merge_asof requires numeric or datetime key; convert trade dates once
        dates_df = pd.DataFrame(
            {"date": pd.to_datetime(trade_dates, format="%Y%m%d", errors="coerce")}
        ).sort_values("date")

        for code in codes:
            stock = (
                fin[fin["code"] == code][["ann_date", col]]
                .rename(columns={"ann_date": "date"})
                .assign(date=lambda d: pd.to_datetime(d["date"], format="%Y%m%d", errors="coerce"))
                .dropna(subset=["date"])
                .sort_values("date")
                .groupby("date")
                .last()
                .reset_index()
            )
            if stock.empty:
                continue
            merged = pd.merge_asof(dates_df, stock, on="date", direction="backward")
            result[code] = merged[col].values

        return result

    # ==================================================================
    # A — Microstructure and price-volume factors
    # ==================================================================

    def factor_amihud_illiq(self) -> pd.DataFrame:
        """
        Amihud (2002) illiquidity: mean(|R_t| / Amount_t) over 20 days.

        Amount is converted from 千元 to 元 (* 1000) before division.
        Result is scaled by 1e6 to bring values into a human-readable range
        (i.e., illiquidity per million yuan of turnover).
        Higher values → less liquid stock.
        """
        if self.amount_raw is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        amount_yuan = self.amount_raw * 1000  # 千元 → 元
        abs_ret = self.returns.abs()
        # Avoid division by zero on zero-volume days
        ratio = abs_ret.div(amount_yuan.replace(0, np.nan))
        return ratio.rolling(20).mean() * 1e6

    def factor_ivol(self) -> pd.DataFrame:
        """
        Idiosyncratic volatility: std of OLS residuals from a 20-day rolling
        regression of individual returns on CSI 300 returns.
        """
        if self.market_ret.empty:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return self._rolling_ols_resid_std(d=20)

    def factor_beta(self) -> pd.DataFrame:
        """
        Rolling 20-day market beta: slope from OLS regression of stock returns
        on CSI 300 returns. Same regression as factor_ivol.
        """
        if self.market_ret.empty:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        beta_df, _ = self._rolling_ols(d=20)
        return beta_df

    def factor_realized_skewness(self) -> pd.DataFrame:
        """
        Realized skewness: bias-corrected skewness of daily returns over 20 days.
        Computed via scipy.stats.skew with bias=False.
        """
        return self._skew(self.returns, d=20)

    def factor_vol_price_corr(self) -> pd.DataFrame:
        """
        Volume-price rank correlation over 10 days.
        Computed as Pearson correlation of cross-sectional ranks, which equals
        Spearman rank correlation.
        """
        return self._corr(self._rank(self.close), self._rank(self.vol), d=10)

    def factor_vol_price_corr_20d(self) -> pd.DataFrame:
        """Volume-price Spearman rank correlation over 20 days."""
        return self._corr(self._rank(self.close), self._rank(self.vol), d=20)

    def factor_vw_return_5d(self) -> pd.DataFrame:
        """
        Volume-weighted return over 5 days:
            sum(R_t * vol_t, 5) / sum(vol_t, 5)
        Measures the average return investors actually received per share traded.
        """
        num = (self.returns * self.vol).rolling(5).sum()
        den = self.vol.rolling(5).sum().replace(0, np.nan)
        return num / den

    def factor_vw_return_10d(self) -> pd.DataFrame:
        """Volume-weighted return over 10 days."""
        num = (self.returns * self.vol).rolling(10).sum()
        den = self.vol.rolling(10).sum().replace(0, np.nan)
        return num / den

    def factor_vol_oscillator(self) -> pd.DataFrame:
        """
        Volume oscillator: VMA5 / VMA20.
        Values > 1 indicate short-term volume surge vs. medium-term baseline.
        """
        vma5  = self.vol.rolling(5).mean()
        vma20 = self.vol.rolling(20).mean().replace(0, np.nan)
        return vma5 / vma20

    def factor_net_buy_proxy_5d(self) -> pd.DataFrame:
        """
        Net buying proxy over 5 days:
            mean( (Close - Open) / (High - Low + 1e-6) * Volume, 5 )
        Approximates directional order flow; positive = net buying pressure.
        """
        direction = (self.close - self.open) / (self.high - self.low + 1e-6)
        return (direction * self.vol).rolling(5).mean()

    def factor_net_buy_proxy_10d(self) -> pd.DataFrame:
        """Net buying proxy over 10 days."""
        direction = (self.close - self.open) / (self.high - self.low + 1e-6)
        return (direction * self.vol).rolling(10).mean()

    def factor_momentum_1d(self) -> pd.DataFrame:
        """1-day price return: (close - delay(close,1)) / delay(close,1)."""
        d1 = self._delay(self.close, 1)
        return (self.close - d1) / d1.replace(0, np.nan)

    def factor_momentum_3d(self) -> pd.DataFrame:
        """3-day cumulative price return."""
        d3 = self._delay(self.close, 3)
        return (self.close - d3) / d3.replace(0, np.nan)

    def factor_momentum_5d(self) -> pd.DataFrame:
        """5-day cumulative price return."""
        d5 = self._delay(self.close, 5)
        return (self.close - d5) / d5.replace(0, np.nan)

    def factor_momentum_10d(self) -> pd.DataFrame:
        """10-day cumulative price return."""
        d10 = self._delay(self.close, 10)
        return (self.close - d10) / d10.replace(0, np.nan)

    def factor_momentum_20d(self) -> pd.DataFrame:
        """20-day cumulative price return."""
        d20 = self._delay(self.close, 20)
        return (self.close - d20) / d20.replace(0, np.nan)

    def factor_trend_strength(self) -> pd.DataFrame:
        """
        Trend strength (Information Ratio proxy) over 20 days:
            cum_ret_20 / sum(|R_t|, 20)
        High values indicate a smooth directional trend; low values indicate noisy
        back-and-forth movement.  Ranges from -1 to +1 in theory.
        """
        cum_ret = self.factor_momentum_20d()
        abs_ret_sum = self._sum(self._abs(self.returns), 20).replace(0, np.nan)
        return cum_ret / abs_ret_sum

    def factor_drawdown_from_high(self) -> pd.DataFrame:
        """
        Distance from 60-day high:
            (close - ts_max(close, 60)) / ts_max(close, 60)
        Always <= 0; captures how far a stock has fallen from its recent peak.
        """
        high60 = self._ts_max(self.close, 60).replace(0, np.nan)
        return (self.close - high60) / high60

    def factor_bias_5d(self) -> pd.DataFrame:
        """
        Price bias from 5-day moving average:
            close / MA(close, 5) - 1
        Positive = price above MA (overbought proxy).
        """
        ma = self.close.rolling(5).mean().replace(0, np.nan)
        return self.close / ma - 1

    def factor_bias_10d(self) -> pd.DataFrame:
        """Price bias from 10-day moving average."""
        ma = self.close.rolling(10).mean().replace(0, np.nan)
        return self.close / ma - 1

    def factor_bias_20d(self) -> pd.DataFrame:
        """Price bias from 20-day moving average."""
        ma = self.close.rolling(20).mean().replace(0, np.nan)
        return self.close / ma - 1

    def factor_rvol_5d(self) -> pd.DataFrame:
        """Realized volatility: rolling 5-day std of daily returns."""
        return self._stddev(self.returns, 5)

    def factor_rvol_10d(self) -> pd.DataFrame:
        """Realized volatility: rolling 10-day std of daily returns."""
        return self._stddev(self.returns, 10)

    def factor_rvol_20d(self) -> pd.DataFrame:
        """Realized volatility: rolling 20-day std of daily returns (total vol, not residual)."""
        return self._stddev(self.returns, 20)

    def factor_realized_kurtosis(self) -> pd.DataFrame:
        """
        Realized excess kurtosis over 20 days (bias-corrected, Fisher definition).
        High positive values indicate fat tails / crash risk.
        """
        return self._kurt(self.returns, d=20)

    def factor_hl_range(self) -> pd.DataFrame:
        """
        Intraday high-low range normalized by close, averaged over 10 days:
            mean( (High - Low) / Close, 10 )
        Proxy for realized intraday volatility and liquidity.
        """
        close_safe = self.close.replace(0, np.nan)
        return ((self.high - self.low) / close_safe).rolling(10).mean()

    def factor_downside_vol(self) -> pd.DataFrame:
        """
        Downside volatility over 20 days: std of negative daily returns only.
        More directly captures left-tail risk than total realized volatility.
        Requires at least 2 negative-return days in the window; otherwise NaN.
        """
        def _dsv(x: np.ndarray) -> float:
            neg = x[x < 0]
            return float(neg.std(ddof=1)) if len(neg) >= 2 else np.nan

        return self.returns.rolling(20).apply(_dsv, raw=True)

    def factor_upside_vol(self) -> pd.DataFrame:
        """
        Upside volatility over 20 days: std of positive daily returns only.
        Complement to factor_downside_vol; captures right-tail dispersion.
        Requires at least 2 positive-return days in the window; otherwise NaN.
        """
        def _usv(x: np.ndarray) -> float:
            pos = x[x > 0]
            return float(pos.std(ddof=1)) if len(pos) >= 2 else np.nan

        return self.returns.rolling(20).apply(_usv, raw=True)

    def factor_gap_return(self) -> pd.DataFrame:
        """
        Overnight gap return: (open_t - close_{t-1}) / close_{t-1}.
        Captures overnight information and pre-market sentiment.
        """
        close_prev = self._delay(self.close, 1)
        return (self.open - close_prev) / close_prev.replace(0, np.nan)

    def factor_turnover_volatility(self) -> pd.DataFrame:
        """
        Rolling 20-day std of turnover_rate.
        High values indicate regime shifts or unusual trading activity.
        """
        if self.turnover_rate is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return self._stddev(self.turnover_rate, 20)

    def factor_distance_from_low(self) -> pd.DataFrame:
        """
        Distance from 60-day low:
            (close - ts_min(close, 60)) / ts_min(close, 60)
        Always >= 0; symmetric to factor_drawdown_from_high.
        """
        low60 = self._ts_min(self.close, 60).replace(0, np.nan)
        return (self.close - low60) / low60

    def factor_momentum_acceleration(self) -> pd.DataFrame:
        """
        Momentum acceleration: momentum_5d - momentum_10d.
        Positive when short-term momentum exceeds medium-term (trend strengthening).
        """
        mom5 = self.factor_momentum_5d()
        mom10 = self.factor_momentum_10d()
        return mom5 - mom10

    def factor_return_consistency(self) -> pd.DataFrame:
        """
        Proportion of positive return days over 20-day rolling window.
        Values in [0, 1]; high values indicate consistent upward drift.
        """
        def _pct_pos(x: np.ndarray) -> float:
            valid = x[~np.isnan(x)]
            if len(valid) < 5:
                return np.nan
            return float(np.mean(valid > 0))

        return self.returns.rolling(20).apply(_pct_pos, raw=True)

    # ==================================================================
    # B — Fundamental and valuation factors
    # ==================================================================

    def factor_ep(self) -> pd.DataFrame:
        """Earnings-to-price ratio: 1 / PE.  NaN when PE is 0 or missing."""
        if self.pe is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return (1.0 / self.pe.replace(0, np.nan))

    def factor_bp(self) -> pd.DataFrame:
        """Book-to-price ratio: 1 / PB.  NaN when PB is 0 or missing."""
        if self.pb is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return (1.0 / self.pb.replace(0, np.nan))

    def factor_roe(self) -> pd.DataFrame:
        """
        ROE (weighted average), PIT-aligned from quarterly fina_indicator.
        Forward-filled from ann_date; updated only when new reports are announced.
        """
        return self._pit_align_financial("roe")

    def factor_log_mv(self) -> pd.DataFrame:
        """
        Log market capitalization: log(total_mv).
        Size factor; smaller stocks typically have higher expected returns.
        """
        return self._log(self.total_mv.replace(0, np.nan))

    def factor_ps(self) -> pd.DataFrame:
        """
        Sales-to-price ratio: 1 / ps_ttm (price-to-sales TTM).
        Valuation factor; higher values indicate cheaper valuation by sales.
        Returns NaN when ps_ttm is missing or zero.
        """
        if self.ps_ttm is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return 1.0 / self.ps_ttm.replace(0, np.nan)

    def factor_dv_ratio(self) -> pd.DataFrame:
        """
        Dividend yield (dv_ratio from daily_basic).
        Higher values indicate higher dividend-paying stocks.
        """
        if self.dv_ratio is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return self.dv_ratio.copy()

    def factor_circ_mv_ratio(self) -> pd.DataFrame:
        """
        Circulating market cap ratio: circ_mv / total_mv.
        Measures float proportion; values in (0, 1].
        """
        if self.circ_mv is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        total_safe = self.total_mv.replace(0, np.nan)
        return self.circ_mv / total_safe

    # ==================================================================
    # C — Cross-sectional relative features
    # ==================================================================

    def factor_industry_rel_turnover(self) -> pd.DataFrame:
        """
        Industry-relative turnover rate:
            turnover_rate − median(turnover_rate in same SW L1 industry)

        Computed cross-sectionally per date.  Industry median uses only
        stocks with valid turnover data on that date.
        """
        if self.turnover_rate is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        return self._industry_demean(self.turnover_rate)

    def factor_industry_rel_bp(self) -> pd.DataFrame:
        """
        Industry-relative book-to-price:
            BP − median(BP in same SW L1 industry)
        """
        if self.pb is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        bp = 1.0 / self.pb.replace(0, np.nan)
        return self._industry_demean(bp)

    def factor_industry_rel_mv(self) -> pd.DataFrame:
        """
        Industry-relative log market cap:
            log(total_mv) − median(log(total_mv) in same SW L1 industry)
        """
        log_mv = self.factor_log_mv()
        return self._industry_demean(log_mv)

    def factor_industry_rel_ep(self) -> pd.DataFrame:
        """
        Industry-relative earnings-to-price:
            EP − median(EP in same SW L1 industry)
        """
        ep = self.factor_ep()
        return self._industry_demean(ep)

    def factor_industry_rel_roe(self) -> pd.DataFrame:
        """
        Industry-relative ROE:
            ROE − median(ROE in same SW L1 industry)
        """
        roe = self.factor_roe()
        return self._industry_demean(roe)

    def factor_ts_rel_turnover(self) -> pd.DataFrame:
        """
        Time-series relative turnover:
            turnover_rate / rolling_mean(turnover_rate, 60)

        Measures how elevated today's turnover is relative to its own history.
        """
        if self.turnover_rate is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        rolling_mean = self.turnover_rate.rolling(60).mean()
        return self.turnover_rate.div(rolling_mean.replace(0, np.nan))

    def factor_ind_rel_momentum_5d(self) -> pd.DataFrame:
        """
        Industry-relative 5-day momentum (difference form):
            momentum_5d − median(momentum_5d in same SW L1 industry)
        Controls for sector-level short-term drift.
        """
        mom = self.factor_momentum_5d()
        return self._industry_demean(mom)

    def factor_ind_rel_momentum_10d(self) -> pd.DataFrame:
        """
        Industry-relative 10-day momentum (difference form):
            momentum_10d − median(momentum_10d in same SW L1 industry)
        """
        mom = self.factor_momentum_10d()
        return self._industry_demean(mom)

    def factor_ind_rel_momentum_20d(self) -> pd.DataFrame:
        """
        Industry-relative 20-day momentum (difference form):
            momentum_20d − median(momentum_20d in same SW L1 industry)
        """
        mom = self.factor_momentum_20d()
        return self._industry_demean(mom)

    def factor_ind_rel_turnover_ratio(self) -> pd.DataFrame:
        """
        Industry-relative turnover ratio (ratio form):
            turnover_rate / mean(turnover_rate in same SW L1 industry)

        Unlike factor_industry_rel_turnover (difference), this ratio form is
        scale-invariant across industries with different average activity levels.
        """
        if self.turnover_rate is None:
            return pd.DataFrame(np.nan, index=self.close.index, columns=self.close.columns)
        industry_map = self._industry.reindex(self.turnover_rate.columns)

        def ratio_row(row: pd.Series) -> pd.Series:
            ind_mean = row.groupby(industry_map).transform("mean")
            return row / ind_mean.replace(0, np.nan)

        return self.turnover_rate.apply(ratio_row, axis=1)

    def factor_ind_rel_vol(self) -> pd.DataFrame:
        """
        Industry-relative realized volatility ratio (20-day):
            rvol_20d / mean(rvol_20d in same SW L1 industry)

        Uses industry mean (not median) as denominator so the ratio is centred
        around 1 rather than 0, making it directly interpretable as a multiplier.
        """
        rvol = self.factor_rvol_20d()
        industry_map = self._industry.reindex(rvol.columns)

        def ratio_row(row: pd.Series) -> pd.Series:
            ind_mean = row.groupby(industry_map).transform("mean")
            return row / ind_mean.replace(0, np.nan)

        return rvol.apply(ratio_row, axis=1)

    def _industry_demean(self, wide: pd.DataFrame) -> pd.DataFrame:
        """
        For each date, subtract the SW L1 industry median from each stock's value.
        Industry membership is static (from stock_info).
        """
        # industry_map: Series indexed by code, values = SW L1 industry string
        industry_map = self._industry.reindex(wide.columns)

        def demean_row(row: pd.Series) -> pd.Series:
            # groupby industry → transform median → subtract
            ind_med = row.groupby(industry_map).transform("median")
            return row - ind_med

        return wide.apply(demean_row, axis=1)

    # ------

    def factor_5_day_reversal(self) -> pd.DataFrame:
        """
        factor_5_day_reversal: (close - delay(close,5)) / delay(close,5)

        Reversal of the last 5 days' closing prices.
        """
        return (self.close - self._delay(self.close, 5)) / self._delay(self.close, 5)   

    def alpha001(self) -> pd.DataFrame:
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)

        For each stock: if today's return is negative, substitute stddev(returns, 20) for
        close; otherwise keep close. Apply SignedPower(x, 2) to the result, then find
        which of the past 5 days had the maximum value (Ts_ArgMax). Cross-sectionally
        rank that position index and subtract 0.5 to centre around zero.
        """
        std20 = self._stddev(self.returns, 20)
        inner = np.where(self.returns < 0, std20, self.close)
        inner = pd.DataFrame(inner, index=self.close.index, columns=self.close.columns)
        sp = self._signed_power(inner, 2.0)
        return self._rank(self._ts_argmax(sp, 5)) - 0.5

    def alpha003(self) -> pd.DataFrame:
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))

        Negative rolling correlation between cross-sectional ranks of open price and
        volume over a 10-day window. Penalises stocks whose open rank and volume rank
        move together (momentum-and-volume agreement is faded).
        """
        return -1 * self._corr(self._rank(self.open), self._rank(self.vol), 10)

    def alpha006(self) -> pd.DataFrame:
        """
        Alpha#6: (-1 * correlation(open, volume, 10))

        Negative time-series correlation between open price and volume
        over a 10-day rolling window, per stock.
        High (positive) correlation is penalized → contrarian tilt.
        """
        return -1 * self._corr(self.open, self.vol, 10)

    def alpha012(self) -> pd.DataFrame:
        """
        Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))

        When volume increases (sign = +1), take a short position in the
        close move (mean-reversion). When volume falls, follow the close move.
        Combines volume direction with intraday price change.
        """
        return self._sign(self._delta(self.vol, 1)) * (-1 * self._delta(self.close, 1))

    def alpha038(self) -> pd.DataFrame:
        """
        Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

        Stocks that are near their recent highs (high ts_rank) AND have a
        high close-to-open ratio are shorted (both ranks are high → product large → -1 applied).
        """
        ts_rnk = self._ts_rank(self.close, 10)
        return (-1 * self._rank(ts_rnk)) * self._rank(self.close / self.open)

    def alpha040(self) -> pd.DataFrame:
        """
        Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

        Stocks with high volatility in the high price (large stddev rank) are shorted
        when high price and volume are positively correlated over the past 10 days.
        Combines dispersion penalty with momentum-volume agreement.
        """
        return (-1 * self._rank(self._stddev(self.high, 10))) * self._corr(self.high, self.vol, 10)

    def alpha041(self) -> pd.DataFrame:
        """
        Alpha#41: (((high * low)^0.5) - vwap)

        Geometric mean of the day's high and low minus vwap.
        Positive when the geometric mean exceeds the typical price;
        may signal upward intraday momentum.
        Uses precise vwap = amount * 10 / vol when available.
        """
        return np.power(self.high * self.low, 0.5) - self.vwap

    def alpha042(self) -> pd.DataFrame:
        """
        Alpha#42: (rank((vwap - close)) / rank((vwap + close)))

        Ratio of the cross-sectional rank of (vwap - close) to the rank of (vwap + close).
        Positive when close < vwap (stock slid below the day's average price); acts as a
        delay-0 mean-reversion signal — stocks that underperformed their vwap are favoured.
        """
        return self._rank(self.vwap - self.close) / self._rank(self.vwap + self.close)

    def alpha054(self) -> pd.DataFrame:
        """
        Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

        ^ denotes signedpower: sign(x)*|x|^e. For positive prices this reduces to
        the ordinary power. Denominator (low - high) <= 0; division by zero yields
        NaN naturally and is left for downstream cleaning.
        """
        num = -1 * (self.low - self.close) * self._signed_power(self.open, 5)
        den = (self.low - self.high) * self._signed_power(self.close, 5)
        return num / den

    def alpha072(self) -> pd.DataFrame:
        """
        Alpha#72: (rank(decay_linear(correlation((high+low)/2, adv40, 8.93345), 10.1519)) /
                   rank(decay_linear(correlation(Ts_Rank(vwap,3.72469),
                                                 Ts_Rank(volume,18.5188), 6.86671), 2.95011)))

        Non-integer window parameters are floored per the paper convention:
          correlation window 8.93345 → 8, decay_linear 10.1519 → 10
          Ts_Rank vwap 3.72469 → 3, Ts_Rank vol 18.5188 → 18
          correlation window 6.86671 → 6, decay_linear 2.95011 → 2
        adv40 approximated as 40-day rolling mean of share volume.
        """
        adv40 = self.vol.rolling(40).mean()
        mid = (self.high + self.low) / 2
        num = self._rank(self._decay_linear(self._corr(mid, adv40, 8), 10))
        ts_rank_vwap = self._ts_rank(self.vwap, 3)
        ts_rank_vol = self._ts_rank(self.vol, 18)
        den = self._rank(self._decay_linear(self._corr(ts_rank_vwap, ts_rank_vol, 6), 2))
        return num / den

    def alpha088(self) -> pd.DataFrame:
        """
        Alpha#88: min(rank(decay_linear(((rank(open)+rank(low))-(rank(high)+rank(close))), 8.06882)),
                      Ts_Rank(decay_linear(correlation(Ts_Rank(close,8.44728),
                                                       Ts_Rank(adv60,20.6966),8.01266),6.65053),2.61957))

        Element-wise minimum of two signals:
          (1) rank of decay-linear-smoothed difference between open/low ranks and high/close ranks
          (2) ts_rank of decay-linear-smoothed rolling correlation between time-series ranks of
              close and adv60
        Non-integer window parameters are floored per paper convention:
          decay_linear 8.06882→8, Ts_Rank close 8.44728→8, Ts_Rank adv60 20.6966→20,
          corr 8.01266→8, decay_linear 6.65053→6, outer Ts_Rank 2.61957→2
        adv60 approximated as 60-day rolling mean of share volume.
        """
        adv60 = self.vol.rolling(60).mean()
        signal1 = self._rank(
            self._decay_linear(
                (self._rank(self.open) + self._rank(self.low))
                - (self._rank(self.high) + self._rank(self.close)),
                8,
            )
        )
        signal2 = self._ts_rank(
            self._decay_linear(
                self._corr(self._ts_rank(self.close, 8), self._ts_rank(adv60, 20), 8),
                6,
            ),
            2,
        )
        return np.minimum(signal1, signal2)

    def alpha094(self) -> pd.DataFrame:
        """
        Alpha#94: ((rank(vwap - ts_min(vwap, 11.5783)) ^
                    Ts_Rank(correlation(Ts_Rank(vwap,19.6462),
                                        Ts_Rank(adv60,4.02992), 18.0926), 2.70756)) * -1)

        ^ denotes signedpower(base, exp).
        Non-integer window parameters are floored per the paper convention:
          ts_min 11.5783 → 11, Ts_Rank vwap 19.6462 → 19
          Ts_Rank adv60 4.02992 → 4, correlation 18.0926 → 18, outer Ts_Rank 2.70756 → 2
        adv60 approximated as 60-day rolling mean of share volume.
        """
        adv60 = self.vol.rolling(60).mean()
        ts_rank_vwap = self._ts_rank(self.vwap, 19)
        ts_rank_adv60 = self._ts_rank(adv60, 4)
        base = self._rank(self.vwap - self._ts_min(self.vwap, 11))
        exp = self._ts_rank(self._corr(ts_rank_vwap, ts_rank_adv60, 18), 2)
        return self._signed_power(base, exp) * -1

    def alpha098(self) -> pd.DataFrame:
        """
        Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5,26.4719), 4.58418), 7.18088)) -
                   rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open),
                                                                    rank(adv15),20.8187),8.62571),
                                             6.95668), 8.07206)))

        Difference of two ranked decay-linear signals:
          (1) rank of decay-smooth correlation between vwap and 26-day sum of adv5
          (2) rank of decay-smooth ts_rank of ts_argmin of correlation between
              open-rank and adv15-rank
        Non-integer window parameters are floored per paper convention:
          sum adv5 26.4719→26, corr 4.58418→4, decay_linear 7.18088→7,
          corr(open,adv15) 20.8187→20, Ts_ArgMin 8.62571→8, Ts_Rank 6.95668→6,
          decay_linear 8.07206→8
        adv5/adv15 approximated as 5/15-day rolling mean of share volume.
        """
        adv5  = self.vol.rolling(5).mean()
        adv15 = self.vol.rolling(15).mean()
        signal1 = self._rank(
            self._decay_linear(
                self._corr(self.vwap, self._sum(adv5, 26), 4),
                7,
            )
        )
        signal2 = self._rank(
            self._decay_linear(
                self._ts_rank(
                    self._ts_argmin(
                        self._corr(self._rank(self.open), self._rank(adv15), 20),
                        8,
                    ),
                    6,
                ),
                8,
            )
        )
        return signal1 - signal2

    def alpha101(self) -> pd.DataFrame:
        """
        Alpha#101: ((close - open) / ((high - low) + .001))

        Intraday momentum: how much the stock closed relative to where it opened,
        normalized by the day's price range. The +0.001 avoids division by zero
        on days with no price movement.
        """
        return (self.close - self.open) / (self.high - self.low + 0.001)

    # ==================================================================
    # Aggregate
    # ==================================================================

    def get_all_factors(self) -> pd.DataFrame:
        """
        Compute every factor_* method and combine into a long-form DataFrame.

        Returns
        -------
        pd.DataFrame
            Index   : MultiIndex (date, code)
            Columns : one per factor, named by the factor_* method name
                      (e.g. 'factor_amihud_illiq', 'factor_ep', ...)
            Values  : raw factor values; NaN and inf preserved for downstream
                      cleaning by FactorCleaner.
        """
        series: dict = {}
        for name in sorted(dir(self)):
            if not (name.startswith("factor_") or name.startswith("alpha")):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            wide = method()
            stacked = wide.stack(dropna=False)
            stacked.name = name
            series[name] = stacked

        if not series:
            return pd.DataFrame()

        result = pd.concat(series, axis=1)
        result.index.names = ["date", "code"]
        return result
