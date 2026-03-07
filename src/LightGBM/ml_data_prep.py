"""
ml_data_prep.py
---------------
Walk-forward time-series splitter for quantitative ML factor research.

Ensures strict temporal ordering so the model never leaks future information
into training or validation windows.

Supports two modes:
- Rolling window (default): fixed-length training window that slides forward.
- Expanding window: training window grows from data start, utilising all
  historical data up to validation start.

Public API
----------
    # Rolling window (fixed train length, slides forward)
    splitter = WalkForwardSplitter(
        train_months=24, val_months=6, test_months=6, embargo_days=5,
        expanding=False,
    )
    for train_mask, val_mask, test_mask in splitter.split(df):
        ...

    # Expanding window (train grows each fold, utilises full history)
    splitter = WalkForwardSplitter(
        train_months=24, val_months=6, test_months=6, embargo_days=5,
        expanding=True,
    )
    for train_mask, val_mask, test_mask in splitter.split(df):
        ...

Parameters
----------
train_months : int
    Length of each training window in calendar months (approximated as
    ``train_months * 21`` trading days).  In expanding mode, this is the
    minimum training length for the first fold.
val_months : int
    Length of the validation window used for early stopping.
test_months : int
    Length of the out-of-sample test window predicted each fold.
embargo_days : int
    Minimum gap (in trading days) between the end of the validation window
    and the start of the test window.  Must be >= the forward-return horizon
    ``d`` to prevent target leakage (default 5, matching ``d=5``).
expanding : bool
    If False (default), use rolling window: train has fixed length and slides.
    If True, use expanding window: train starts at data start and grows each
    fold, utilising all history up to validation start.
step_months : int, optional
    Forward step in months per fold.  If None (default), step = test_months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_MONTH: int = 21


class WalkForwardSplitter:
    """Quantitative-finance walk-forward cross-validator.

    Each call to :meth:`split` yields boolean masks for train / validation /
    test subsets derived from the unique sorted trading dates in *df*.

    Two modes:
    - Rolling (expanding=False): fixed-length train window slides forward.
    - Expanding (expanding=True): train starts at data start and grows each
      fold, utilising all history up to validation start.

    Parameters
    ----------
    train_months : int
        Training window length in months (default 24).  In expanding mode,
        minimum train length for the first fold.
    val_months : int
        Validation window length in months (default 6).
    test_months : int
        Test window length in months (default 6).
    embargo_days : int
        Trading-day gap between end of validation and start of test to avoid
        target leakage (default 5).
    expanding : bool
        If True, use expanding window; if False, use rolling window (default).
    step_months : int, optional
        Forward step in months per fold.  If None, step = test_months.
    """

    def __init__(
        self,
        train_months: int = 24,
        val_months: int = 6,
        test_months: int = 6,
        embargo_days: int = 5,
        expanding: bool = False,
        step_months: int | None = None,
    ) -> None:
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.embargo_days = embargo_days
        self.expanding = expanding
        self.step_months = step_months if step_months is not None else test_months

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame):
        """Yield boolean masks (train, val, test) for each walk-forward fold.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a column named ``trade_date`` (YYYYMMDD int or
            datetime-like).  Rows need not be pre-sorted.

        Yields
        ------
        train_mask : pd.Series[bool]
        val_mask   : pd.Series[bool]
        test_mask  : pd.Series[bool]
        """
        if "trade_date" not in df.columns:
            raise ValueError(
                "df must contain a 'trade_date' column; "
                f"found columns: {list(df.columns)}"
            )

        dates: np.ndarray = np.sort(df["trade_date"].unique())
        total_len = len(dates)

        train_len = self.train_months * TRADING_DAYS_PER_MONTH
        val_len = self.val_months * TRADING_DAYS_PER_MONTH
        test_len = self.test_months * TRADING_DAYS_PER_MONTH
        step_len = self.step_months * TRADING_DAYS_PER_MONTH

        start_idx = 0

        while True:
            val_start = start_idx + train_len
            val_end = val_start + val_len
            test_start = val_end + self.embargo_days
            test_end = test_start + test_len

            if test_end > total_len:
                break

            if self.expanding:
                train_start = 0
            else:
                train_start = start_idx

            train_dates = dates[train_start:val_start]
            val_dates = dates[val_start:val_end]
            test_dates = dates[test_start:test_end]

            train_mask = df["trade_date"].isin(train_dates)
            val_mask = df["trade_date"].isin(val_dates)
            test_mask = df["trade_date"].isin(test_dates)

            yield train_mask, val_mask, test_mask

            start_idx += step_len

    def n_splits(self, df: pd.DataFrame) -> int:
        """Return the total number of folds without materialising masks."""
        dates = np.sort(df["trade_date"].unique())
        total_len = len(dates)
        train_len = self.train_months * TRADING_DAYS_PER_MONTH
        val_len = self.val_months * TRADING_DAYS_PER_MONTH
        test_len = self.test_months * TRADING_DAYS_PER_MONTH
        step_len = self.step_months * TRADING_DAYS_PER_MONTH
        min_required = train_len + val_len + self.embargo_days + test_len
        if total_len < min_required:
            return 0
        return (total_len - min_required) // step_len + 1
