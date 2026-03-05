"""
optimizer.py
------------
Convex portfolio optimiser for the long-only, industry-neutral strategy.

Problem (per trading day t)
----------------------------
    max_{w}  w' alpha_centered  -  lambda_turnover/2 * ||w - w_prev||_1

    s.t.  sum(w) = 1                           (fully invested)
          w >= 0                                (long only)
          w <= max_weight                       (single-stock cap)
          |X_ind' w - w_bench| <= tol           (industry neutrality)

where
    w              : portfolio weights, shape (n,)
    alpha_centered : cross-sectionally de-meaned alpha signal, shape (n,)
                     Caller is responsible for de-meaning before calling solve().
                     Typical range after de-meaning: [-0.5, 0.5]
    w_prev         : previous-day weights, shape (n,)
    X_ind          : industry dummy matrix, shape (n, K)
    w_bench        : benchmark industry weights, shape (K,)
    lambda_turnover: turnover-aversion coefficient in the objective (default 0.2).
                     This is NOT a transaction cost rate; it is a dimensionless
                     policy parameter controlling the trade-off between alpha
                     capture and portfolio stability.  See parameter docs below.
    tol            : industry deviation tolerance (default ±0.01)

Two-parameter design
---------------------
This module uses ``lambda_turnover`` solely as a signal-vs-stability trade-off
knob.  The actual monetary cost of trading (0.2 % per unit turnover) is
accounted for separately in ``OptimizationBacktester._build_return_series()``
and must NOT be conflated with this parameter.

Why separate?
  - ``alpha_centered`` is in rank-score units (roughly ±0.5), NOT in return
    units (roughly ±0.01).  A cost rate of 0.002 would be ~250× too small
    to have any effect on the optimiser.
  - ``lambda_turnover`` is calibrated empirically by inspecting the reported
    ``Avg Daily Turnover`` in the backtest summary.  Typical guidance:
      * lambda = 0.05~0.1  → high turnover (~10–20 % daily)
      * lambda = 0.2~0.5   → moderate turnover (~2–8 % daily)  ← recommended
      * lambda = 1.0+      → very low turnover, slow signal tracking

Infeasibility handling
-----------------------
When the industry constraint makes the problem infeasible (e.g. too few tradable
stocks in some sectors), ``tol`` is automatically relaxed in steps of
``industry_tol_step`` up to ``industry_tol_max``.  The actual tolerance used is
returned alongside the optimal weights.  If even the maximum tolerance yields no
feasible solution, the industry constraint is dropped entirely as a last resort
and a warning is printed.

Public API
----------
    opt = PortfolioOptimizer(lambda_turnover=0.2, max_weight=0.05,
                             industry_tol=0.01, industry_tol_max=0.05,
                             industry_tol_step=0.01)
    w_star, tol_used = opt.solve(alpha_centered_t, w_prev, X_industry, w_benchmark)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "cvxpy is required for portfolio optimisation. "
        "Install it with: pip install cvxpy"
    ) from exc


class PortfolioOptimizer:
    """Convex LP-based portfolio optimiser with automatic industry-tol relaxation.

    Parameters
    ----------
    lambda_turnover : float
        Turnover-aversion coefficient in the optimiser objective (default 0.2).

        **This is NOT the transaction cost rate.**  It is a dimensionless
        policy parameter that controls how much the optimiser penalises
        deviations from the previous-day weights relative to alpha capture.

        The objective term is:  ``lambda_turnover * 0.5 * ||w - w_prev||_1``

        Because the input alpha is in cross-sectional rank-score units
        (roughly ±0.5 after de-meaning) rather than return units (±0.01),
        a cost rate of 0.002 would be ~250× too small to have meaningful
        effect.  This parameter should instead be tuned by inspecting the
        ``Avg Daily Turnover`` in the backtest report:
          - lambda = 0.05~0.1  → aggressive tracking, ~10–20 % daily turnover
          - lambda = 0.2~0.5   → moderate stability, ~2–8 % daily turnover
          - lambda = 1.0+      → slow/stable portfolio

        The actual P&L deduction for trading costs (0.2 % of turnover) is
        handled separately in OptimizationBacktester and is independent of
        this parameter.
    max_weight : float
        Maximum weight allowed for any single stock (default 0.05 = 5%).
    industry_tol : float
        Initial allowed absolute deviation of portfolio industry weights from
        the benchmark (default 0.01 = ±1 pp).
    industry_tol_max : float
        Upper bound on the relaxed tolerance (default 0.05 = ±5 pp).
    industry_tol_step : float
        Step size for relaxing the tolerance on each retry (default 0.01).
    solver : str or None
        cvxpy solver name.  None lets cvxpy choose (typically CLARABEL for LP).
    """

    def __init__(
        self,
        lambda_turnover: float = 0.2,
        max_weight: float = 0.05,
        industry_tol: float = 0.01,
        industry_tol_max: float = 0.05,
        industry_tol_step: float = 0.01,
        solver: Optional[str] = None,
    ) -> None:
        self.lambda_turnover = lambda_turnover
        self.max_weight = max_weight
        self.industry_tol = industry_tol
        self.industry_tol_max = industry_tol_max
        self.industry_tol_step = industry_tol_step
        self.solver = solver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        alpha_t: np.ndarray,
        w_prev: np.ndarray,
        X_industry: np.ndarray,
        w_benchmark: np.ndarray,
    ) -> tuple[np.ndarray, Optional[float]]:
        """Solve the portfolio optimisation problem for a single day.

        Parameters
        ----------
        alpha_t : np.ndarray, shape (n,)
            Cross-sectionally de-meaned alpha signal for each stock in the
            tradable universe.  Must already have the cross-sectional mean
            subtracted (i.e. alpha_t = raw_alpha - mean(raw_alpha)) so that
            the signal is centred around 0.  Typical range: [-0.5, 0.5].
            De-meaning is performed by the caller (OptimizationBacktester)
            to keep this class stateless with respect to the universe.
        w_prev : np.ndarray, shape (n,)
            Previous-day portfolio weights for the same universe.
            New-to-universe stocks should be initialised to 0.
        X_industry : np.ndarray, shape (n, K)
            Binary industry dummy matrix.
        w_benchmark : np.ndarray, shape (K,)
            Benchmark industry weight vector (sums to 1).

        Returns
        -------
        w_star : np.ndarray, shape (n,)
            Optimal portfolio weights.  Sum to 1, all non-negative.
        tol_used : float or None
            Industry tolerance actually used.  Equal to ``industry_tol`` when
            the problem solved on the first try; larger if relaxed; None if the
            industry constraint was dropped as a last resort.
        """
        n = len(alpha_t)

        # Handle edge cases
        if n == 0:
            return np.array([]), None
        if n == 1:
            return np.array([1.0]), self.industry_tol

        alpha_t  = np.asarray(alpha_t,  dtype=float)
        w_prev   = np.asarray(w_prev,   dtype=float)
        X_ind    = np.asarray(X_industry, dtype=float)
        w_bench  = np.asarray(w_benchmark, dtype=float)

        w = cp.Variable(n, nonneg=True)

        # Objective: maximise alpha return minus turnover-aversion penalty.
        # lambda_turnover controls the signal-vs-stability trade-off;
        # it is NOT a monetary cost rate (see class docstring).
        objective = cp.Maximize(
            w @ alpha_t - self.lambda_turnover * 0.5 * cp.norm1(w - w_prev)
        )

        # Base constraints (never relaxed)
        base_constraints = [
            cp.sum(w) == 1,
            w <= self.max_weight,
        ]

        # Try progressively relaxed industry tolerances
        tol_values = np.round(
            np.arange(
                self.industry_tol,
                self.industry_tol_max + self.industry_tol_step / 2,
                self.industry_tol_step,
            ),
            decimals=4,
        )

        for tol in tol_values:
            industry_constraints = [
                X_ind.T @ w - w_bench >= -tol,
                X_ind.T @ w - w_bench <=  tol,
            ]
            problem = cp.Problem(objective, base_constraints + industry_constraints)
            self._solve_problem(problem)
            if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return np.clip(w.value, 0.0, None), float(tol)

        # Last resort: drop industry constraint entirely
        warnings.warn(
            f"Industry constraint infeasible up to tol={self.industry_tol_max:.2%}. "
            "Solving without industry constraint.",
            RuntimeWarning,
            stacklevel=2,
        )
        problem_no_ind = cp.Problem(objective, base_constraints)
        self._solve_problem(problem_no_ind)
        if problem_no_ind.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.clip(w.value, 0.0, None), None

        # Absolute fallback: equal weight
        warnings.warn(
            "Optimisation failed entirely. Falling back to equal weights.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(n, 1.0 / n), None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solve_problem(self, problem: "cp.Problem") -> None:
        """Invoke the cvxpy solver, suppressing verbose output."""
        solver_kwargs: dict = {"verbose": False}
        if self.solver is not None:
            solver_kwargs["solver"] = self.solver
        try:
            problem.solve(**solver_kwargs)
        except cp.SolverError:
            pass
