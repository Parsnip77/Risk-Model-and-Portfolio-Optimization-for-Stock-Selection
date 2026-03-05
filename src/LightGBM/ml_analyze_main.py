"""
ml_analyze_main.py
------------------
Stage 2 pipeline: LightGBM alpha synthesis + IC analysis + backtest + report.

Workflow
--------
1.  Load ``factors_clean.parquet``, ``prices.parquet``, ``meta.parquet``.
2.  Compute T+d raw forward returns via ``calc_forward_return``.
3.  Compute industry-neutralized forward returns:
        ind_neutral_return = forward_return - mean(forward_return in same SW L1 industry)
    Stocks whose industry has only one tradable member on a given day receive NaN
    and are excluded automatically via dropna.
4.  Merge factors + ind_neutral_return + raw forward_return; drop NaN rows.
    Training target : cs_rank_return = pct_rank(ind_neutral_return) per trade_date.
    Backtest / IC   : raw forward_return (unmodified).
5.  Initialise WalkForwardSplitter (train=18m, val=3m, test=3m, embargo=1d).
6.  Per fold: train AlphaLGBM on cs_rank_return, predict test set, record predictions.
7.  Concatenate all fold predictions; apply 3-day rolling-mean smoothing.
8.  IC analysis (Spearman IC vs raw forward_return): IC mean / std / ICIR, IC chart.
9.  Industry-neutral layered backtest (Plan B: industry-equal-weight) → NAV chart.
10. Industry-neutral net return backtest (Plan B long-only) → NAV + cost metrics.
11. Average feature importance across folds → bar chart (plots/feature_importance.png).
12. SHAP beeswarm on last-fold test sample, top-10 features (plots/shap_beeswarm.png).
13. Write text report to ``result_ml.txt``.

Usage
-----
    python src/LightGBM/ml_analyze_main.py
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
import warnings
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project module imports
# ---------------------------------------------------------------------------
# __file__ is src/LightGBM/ml_analyze_main.py → project root is two levels up
_ROOT = pathlib.Path(__file__).parent.parent.parent
_LGBM_DIR = pathlib.Path(__file__).parent          # src/LightGBM/
sys.path.insert(0, str(_LGBM_DIR))
sys.path.insert(0, str(_ROOT / "src" / "portfolio"))

from targets import calc_forward_return          
from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic  
from backtester import LayeredBacktester         
from net_backtester import NetReturnBacktester   
from ml_data_prep import WalkForwardSplitter     
from lgbm_model import AlphaLGBM                

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# All output paths are anchored to the project root so the script can be
# invoked from any working directory.
DATA_DIR    = _ROOT / "data"
PLOTS_DIR   = _ROOT / "plots"
RESULT_FILE = _ROOT / "result_ml.txt"

FORWARD_DAYS: int = 1
TRAIN_MONTHS: int = 18
VAL_MONTHS: int = 3
TEST_MONTHS: int = 3
EMBARGO_DAYS: int = 1        # must be >= FORWARD_DAYS to prevent target leakage
SHAP_SAMPLE_SIZE: int = 300  # rows sub-sampled for SHAP (speed)
SHAP_TOP_N: int = 10         # features displayed in beeswarm

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print(msg: str, buf: StringIO) -> None:
    """Echo to stdout and accumulate into the string buffer."""
    print(msg)
    buf.write(msg + "\n")


def _section(title: str, buf: StringIO, width: int = 70) -> None:
    _print("\n" + "=" * width, buf)
    _print(f"  {title}", buf)
    _print("=" * width, buf)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def _compute_industry_neutral_return(
    raw_ret_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Subtract the same-day, same-industry mean return from each stock's return.

    Parameters
    ----------
    raw_ret_df : pd.DataFrame
        Flat table with columns [trade_date, ts_code, forward_return].
    meta_df : pd.DataFrame
        Flat table with columns [trade_date, ts_code, industry, ...].

    Returns
    -------
    pd.DataFrame
        Same schema as raw_ret_df, with an additional column
        ``ind_neutral_return``.  Stocks that are the sole member of their
        industry on a given date receive NaN (excluded by downstream dropna).
    """
    ind_map = meta_df[["trade_date", "ts_code", "industry"]].drop_duplicates()
    df = raw_ret_df.merge(ind_map, on=["trade_date", "ts_code"], how="left")

    # Industry mean: only meaningful when >= 2 members are present on that date
    def _ind_mean(grp: pd.Series) -> pd.Series:
        n_valid = grp.notna().sum()
        mean_val = grp.mean() if n_valid >= 2 else np.nan
        return grp - mean_val

    df["ind_neutral_return"] = (
        df.groupby(["trade_date", "industry"])["forward_return"]
        .transform(_ind_mean)
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    report_buf = StringIO()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    _section("Stage 2 · LightGBM Alpha Synthesis", report_buf)

    factors_path = DATA_DIR / "factors_clean.parquet"
    prices_path  = DATA_DIR / "prices.parquet"
    meta_path    = DATA_DIR / "meta.parquet"

    for p in (factors_path, prices_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run src/data_preparation/data_preparation_main.py first."
            )

    _print("1. Loading parquet files ...", report_buf)
    factors_df = pd.read_parquet(factors_path)
    prices_df  = pd.read_parquet(prices_path)
    meta_df    = pd.read_parquet(meta_path)

    # Flatten MultiIndex if present
    for df in (factors_df, prices_df, meta_df):
        if isinstance(df.index, pd.MultiIndex):
            df.reset_index(inplace=True)

    _print(f"   factors shape : {factors_df.shape}", report_buf)
    _print(f"   prices  shape : {prices_df.shape}", report_buf)
    _print(f"   meta    shape : {meta_df.shape}", report_buf)

    # -----------------------------------------------------------------------
    # 2. Raw forward returns
    # -----------------------------------------------------------------------
    _print(f"\n2. Computing {FORWARD_DAYS}-day raw forward returns ...", report_buf)
    target_df  = calc_forward_return(prices_df, d=FORWARD_DAYS)
    target_flat = target_df.reset_index()[["trade_date", "ts_code", "forward_return"]]

    # -----------------------------------------------------------------------
    # 3. Industry-neutralized returns
    # -----------------------------------------------------------------------
    _print("\n3. Computing industry-neutralized returns ...", report_buf)
    target_with_neutral = _compute_industry_neutral_return(target_flat, meta_df)
    neutral_flat = target_with_neutral[
        ["trade_date", "ts_code", "forward_return", "ind_neutral_return"]
    ]

    # -----------------------------------------------------------------------
    # 4. Merge factors + targets
    # -----------------------------------------------------------------------
    _print("\n4. Merging factors and targets ...", report_buf)

    key_cols     = {"trade_date", "ts_code"}
    feature_cols = [c for c in factors_df.columns if c not in key_cols]

    df_merged = pd.merge(
        factors_df[["trade_date", "ts_code"] + feature_cols],
        neutral_flat,
        on=["trade_date", "ts_code"],
        how="inner",
    ).dropna(subset=feature_cols + ["ind_neutral_return", "forward_return"])

    df_merged = df_merged.sort_values("trade_date").reset_index(drop=True)

    # Training target: cross-sectional pct-rank of the industry-neutral return.
    # This removes both market beta and industry beta from the label, so the
    # model learns pure stock-selection signal within each sector.
    df_merged["cs_rank_return"] = df_merged.groupby("trade_date")[
        "ind_neutral_return"
    ].rank(pct=True)

    _print(f"   merged shape          : {df_merged.shape}", report_buf)
    _print(f"   feature cols ({len(feature_cols)}): {feature_cols[:5]}...", report_buf)
    _print(
        "   training target       : cs_rank_return  "
        "(cross-sectional pct-rank of industry-neutral T+1 return)",
        report_buf,
    )
    _print(
        "   IC / backtest target  : forward_return  "
        "(raw, unmodified)",
        report_buf,
    )

    # -----------------------------------------------------------------------
    # 5. Walk-forward cross-validation
    # -----------------------------------------------------------------------
    _section("Walk-Forward Training", report_buf)

    splitter = WalkForwardSplitter(
        train_months=TRAIN_MONTHS,
        val_months=VAL_MONTHS,
        test_months=TEST_MONTHS,
        embargo_days=EMBARGO_DAYS,
    )

    estimated_folds = splitter.n_splits(df_merged)
    _print(
        f"   Settings : train={TRAIN_MONTHS}m  val={VAL_MONTHS}m  "
        f"test={TEST_MONTHS}m  embargo={EMBARGO_DAYS}d",
        report_buf,
    )
    _print(f"   Estimated folds : {estimated_folds}", report_buf)

    if estimated_folds == 0:
        raise RuntimeError(
            "Not enough data for even one fold. "
            "Consider reducing train_months / val_months / test_months."
        )

    all_predictions: list[pd.DataFrame] = []
    fold_importances: list[pd.DataFrame] = []
    last_X_test: pd.DataFrame | None = None
    last_model: AlphaLGBM | None = None

    for fold_idx, (train_mask, val_mask, test_mask) in enumerate(
        splitter.split(df_merged), start=1
    ):
        df_train = df_merged[train_mask]
        df_val   = df_merged[val_mask]
        df_test  = df_merged[test_mask]

        X_train = df_train[feature_cols]
        y_train = df_train["cs_rank_return"]
        X_val   = df_val[feature_cols]
        y_val   = df_val["cs_rank_return"]
        X_test  = df_test[feature_cols]

        _print(
            f"\n--- Fold {fold_idx} ---\n"
            f"  Train : {df_train['trade_date'].min()} → {df_train['trade_date'].max()}"
            f"  (n={len(X_train):,})\n"
            f"  Val   : {df_val['trade_date'].min()} → {df_val['trade_date'].max()}"
            f"  (n={len(X_val):,})\n"
            f"  Test  : {df_test['trade_date'].min()} → {df_test['trade_date'].max()}"
            f"  (n={len(X_test):,})",
            report_buf,
        )

        ml_model = AlphaLGBM()
        ml_model.train(X_train, y_train, X_val, y_val)

        best_iter = ml_model.model.best_iteration_
        _print(f"  Best iteration : {best_iter}", report_buf)

        y_pred = ml_model.predict(X_test)

        df_pred_chunk = df_test[["trade_date", "ts_code"]].copy()
        df_pred_chunk["ml_alpha"] = y_pred
        all_predictions.append(df_pred_chunk)

        fold_importances.append(ml_model.get_feature_importance())

        last_X_test = X_test
        last_model  = ml_model

    # -----------------------------------------------------------------------
    # 7. Assemble final alpha with 3-day rolling mean smoothing
    # -----------------------------------------------------------------------
    _section("Assembling Final ML Alpha", report_buf)

    final_alpha_df = pd.concat(all_predictions, ignore_index=True)

    # 3-day rolling mean per stock: smooths noise and lowers turnover.
    # Stocks with fewer than 3 history points in the prediction window → NaN.
    final_alpha_df = final_alpha_df.sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)
    final_alpha_df["ml_alpha"] = (
        final_alpha_df.groupby("ts_code")["ml_alpha"]
        .transform(lambda s: s.rolling(window=1, min_periods=1).mean())
    )
    final_alpha_df = final_alpha_df.dropna(subset=["ml_alpha"]).reset_index(drop=True)

    _print(f"   shape      : {final_alpha_df.shape}  (after 3-day smoothing)", report_buf)
    _print(
        f"   date range : {final_alpha_df['trade_date'].min()} → "
        f"{final_alpha_df['trade_date'].max()}",
        report_buf,
    )

    # Persist alpha signal for downstream optimization backtester
    alpha_parquet_path = DATA_DIR / "ml_alpha.parquet"
    final_alpha_df.to_parquet(alpha_parquet_path, index=False)
    _print(f"   Saved      : {alpha_parquet_path}", report_buf)

    # -----------------------------------------------------------------------
    # 8. IC analysis  (vs raw forward_return)
    # -----------------------------------------------------------------------
    _section("IC Analysis — ML Synthetic Factor", report_buf)

    ic_series  = calc_ic(final_alpha_df, target_flat)
    ic_metrics = calc_ic_metrics(ic_series)

    _print(f"   IC Mean : {ic_metrics['ic_mean']:>+.4f}", report_buf)
    _print(f"   IC Std  : {ic_metrics['ic_std']:>.4f}", report_buf)
    _print(f"   ICIR    : {ic_metrics['icir']:>+.4f}", report_buf)

    ic_path = PLOTS_DIR / "ml_alpha_ic.png"
    plot_ic(ic_series, factor_name="ml_alpha", show=False, save_path=ic_path)
    plt.close("all")
    _print(f"\n   IC chart saved : {ic_path}", report_buf)

    # -----------------------------------------------------------------------
    # 9. Layered backtest  (industry-neutral, vs raw forward_return)
    # -----------------------------------------------------------------------
    _section("Layered Backtest (LayeredBacktester, Industry-Neutral)", report_buf)

    industry_df = meta_df[["trade_date", "ts_code", "industry"]].drop_duplicates()

    bt = LayeredBacktester(
        final_alpha_df,
        target_flat,
        industry_df=industry_df,
        num_groups=5,
        rf=0.03,
        forward_days=FORWARD_DAYS,
        plots_dir=PLOTS_DIR,
    )
    perf_table = bt.run_backtest()
    bt.plot(show=False)

    _print("\nLayered Backtest Performance (Industry-Neutral):\n", report_buf)
    _print(perf_table.to_string(), report_buf)

    # -----------------------------------------------------------------------
    # 10. Net return backtest  (industry-neutral long-only, Plan B)
    # -----------------------------------------------------------------------
    _section("Net Return Backtest (NetReturnBacktester, Industry-Neutral)", report_buf)

    nb = NetReturnBacktester(
        final_alpha_df,
        prices_df,
        industry_df=industry_df,
        forward_days=FORWARD_DAYS,
        cost_rate=0.002,
        rf=0.03,
        plots_dir=PLOTS_DIR,
    )
    net_summary = nb.run_backtest()
    nb.plot(show=False)

    _print("\nNet Return Backtest Summary:\n", report_buf)
    _print(net_summary.to_string(), report_buf)

    # -----------------------------------------------------------------------
    # 11. Average feature importance across folds
    # -----------------------------------------------------------------------
    _section("Feature Importance (Average Across Folds)", report_buf)

    avg_importance = (
        pd.concat(fold_importances, ignore_index=True)
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    _print("\nAverage Feature Importance (gain):\n", report_buf)
    _print(avg_importance.to_string(index=False), report_buf)

    fig_imp, ax_imp = plt.subplots(figsize=(8, max(4, len(avg_importance) * 0.45)))
    ax_imp.barh(
        avg_importance["feature"][::-1],
        avg_importance["importance"][::-1],
    )
    ax_imp.set_title(f"Average Feature Importance (gain) — {estimated_folds} Folds")
    ax_imp.set_xlabel("Mean Importance (gain)")
    fig_imp.tight_layout()
    imp_path = PLOTS_DIR / "feature_importance.png"
    fig_imp.savefig(imp_path, dpi=150)
    plt.close(fig_imp)
    _print(f"\n   Saved : {imp_path}", report_buf)

    # -----------------------------------------------------------------------
    # 12. SHAP beeswarm (last fold, top-10 features)
    # -----------------------------------------------------------------------
    _section("SHAP Analysis (Last Fold, Top-10 Features)", report_buf)

    if last_X_test is not None and last_model is not None:
        sample_size = min(SHAP_SAMPLE_SIZE, len(last_X_test))
        X_shap = last_X_test.sample(n=sample_size, random_state=42)
        _print(f"   SHAP sample size : {sample_size} rows", report_buf)
        try:
            shap_path = PLOTS_DIR / "shap_beeswarm.png"
            last_model.plot_shap(X_shap, save_path=shap_path, max_display=SHAP_TOP_N)
            plt.close("all")
            _print(f"   Saved : {shap_path}", report_buf)
        except ImportError as e:
            _print(f"   SHAP skipped : {e}", report_buf)

    # -----------------------------------------------------------------------
    # 13. Write report
    # -----------------------------------------------------------------------
    _section("Report Written", report_buf)
    report_text = report_buf.getvalue()
    RESULT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to : {RESULT_FILE}")
    print("Charts saved to      : plots/")


if __name__ == "__main__":
    main()
