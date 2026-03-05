"""
targets.py
----------
Label generation module for forward returns.
"""

import pandas as pd


def calc_forward_return(prices_df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Calculate d-day forward return aligned to time T.

    Execution assumption (no look-ahead bias)
    ------------------------------------------
    Signal is computed from data available at the close of day T.
    The trade is submitted overnight and executes at the OPEN of day T+1.
    The position is then held until the OPEN of day T+d+1.

        forward_return_T = open_{T+d+1} / open_{T+1} - 1

    For the default d=1 this becomes open_{T+2} / open_{T+1} - 1, i.e. one
    open-to-open trading day.  This is consistent with what both
    NetReturnBacktester and OptimizationBacktester actually realise, since
    those modules also use open-to-open daily returns.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Flat DataFrame with columns [trade_date, ts_code, open, close, ...].
    d : int
        Number of trading days to hold (open_{T+1} → open_{T+d+1}).

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with MultiIndex (trade_date, ts_code) and a single
        column ``forward_return`` = open_{T+d+1} / open_{T+1} - 1.
        Rows for which T+d+1 falls outside the data range contain NaN.
    """
    open_wide = prices_df.pivot(index="trade_date", columns="ts_code", values="open")

    # open_wide.shift(-1)      at row T  →  open_{T+1}   (entry price)
    # open_wide.shift(-(d+1))  at row T  →  open_{T+d+1} (exit price)
    fwd_wide = open_wide.shift(-(d + 1)) / open_wide.shift(-1) - 1

    # Stack back to long form, preserving NaN for boundary dates
    fwd_long = (
        fwd_wide.stack(future_stack=True)
        .rename("forward_return")
        .reset_index()
    )
    fwd_long = fwd_long.set_index(["trade_date", "ts_code"])

    return fwd_long
