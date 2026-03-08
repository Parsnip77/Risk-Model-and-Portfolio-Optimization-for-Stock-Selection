"""
optimizer.py
------------
Convex portfolio optimiser for the long-only, industry-neutral strategy.

Problem (per trading day t)
----------------------------
    max_{w}  w' alpha_centered  -  lambda_turnover/2 * ||w - w_prev||_1
             - 1/2 * mu_risk * w^T Σ w                (optional risk penalty)

    s.t.  sum(w) = 1                           (fully invested)
          w >= 0                                (long only)
          w <= max_weight                       (single-stock cap)
          0.5||w - w_prev||_1 <= max_turnover   (daily turnover cap, if set)
          |X_ind' w - w_bench| <= tol           (industry neutrality)
          1/2 w^T Σ w <= max_variance           (optional risk limit)

where
    w              : portfolio weights, shape (n,)
    alpha_centered : cross-sectionally de-meaned alpha signal, shape (n,)
                     Caller is responsible for de-meaning before calling solve().
                     Typical range after de-meaning: [-0.5, 0.5]
    w_prev         : previous-day weights, shape (n,)
    X_ind          : industry dummy matrix, shape (n, K_ind)
    w_bench        : benchmark industry weights, shape (K_ind,)
    lambda_turnover: turnover-aversion coefficient in the objective (default 0.2).
                     This is NOT a transaction cost rate; it is a dimensionless
                     policy parameter controlling the trade-off between alpha
                     capture and portfolio stability.  See parameter docs below.
    tol            : industry deviation tolerance (default ±0.01)
    Σ              : covariance matrix estimated from multi-factor risk model.
                     Decomposed as Σ = X_risk F X_risk^T + Δ for efficiency.

Risk model integration (optional)
----------------------------------
When X_risk, F_half, delta_std are passed to solve(), the quadratic risk term
is computed without constructing the full N×N matrix:

    w^T Σ w = ||F_half @ (X_risk^T w)||² + ||delta_std ⊙ w||²

where
    X_risk   : (n, K) factor exposure matrix (style + industry, from risk model)
    F_half   : (K, K) upper Cholesky factor of F_t  (F = F_half.T @ F_half)
    delta_std: (n,) per-stock idiosyncratic standard deviation (sqrt of Δ_{ii})

This decomposition enables cvxpy to represent the risk term as a sum of
squared norms (SOCP), which CLARABEL solves natively and efficiently.

mu_risk calibration
--------------------
alpha is in rank-score units (~±0.5 after de-meaning).
risk_quad (w^T Σ w) is in daily-variance units (~0.0001 for a diversified portfolio).
To make the risk penalty comparable to the alpha term:
    mu_risk ≈ alpha_scale / risk_scale ≈ 0.5 / 0.0001 = 5000
Recommended starting range: 1000–10000.
Inspect 'Avg Daily Variance' in the backtest report to calibrate.

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
                             max_turnover=0.10, industry_tol=0.01,
                             mu_risk=0.0, max_variance=None)
    w_star, tol_used, fallback_used = opt.solve(alpha_t, w_prev, X_industry,
                                                w_benchmark, X_risk=X_t,
                                                F_half=L_T, delta_std=delta_t)
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
    """Convex portfolio optimiser (LP/SOCP) with automatic industry-tol relaxation.

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
    max_turnover : float or None
        Hard cap on daily one-way turnover: 0.5||w - w_prev||_1 <= max_turnover.
        Default 0.10 = 10% daily turnover limit.  None disables this constraint.
    industry_tol : float
        Initial allowed absolute deviation of portfolio industry weights from
        the benchmark (default 0.01 = ±1 pp).
    industry_tol_max : float
        Upper bound on the relaxed tolerance (default 0.05 = ±5 pp).
    industry_tol_step : float
        Step size for relaxing the tolerance on each retry (default 0.01).
    mu_risk : float
        Risk-aversion coefficient for the quadratic risk penalty (default 0.0).
        Adds term: -0.5 * mu_risk * w^T Σ w to the objective.
        Set to 0.0 to disable.  Typical calibration range: 1000–10000.
        Only active when X_risk, F_half, delta_std are passed to solve().
        See module docstring for calibration guidance.
    max_variance : float or None
        Hard upper bound on daily portfolio variance: w^T Σ w <= 2*max_variance.
        None (default) disables this constraint.
        Only active when X_risk, F_half, delta_std are passed to solve().
    solver : str or None
        cvxpy solver name.  None lets cvxpy choose (CLARABEL supports LP and SOCP).
    style_tol : float
        Absolute tolerance for style factor neutrality: |w_active' X_factor| <= style_tol.
        Only active when w_benchmark_stock and X_style are passed to solve().
    """

    def __init__(
        self,
        lambda_turnover: float = 0.2,
        max_weight: float = 0.05,
        industry_tol: float = 0.01,
        industry_tol_max: float = 0.05,
        industry_tol_step: float = 0.01,
        max_turnover: Optional[float] = 0.10,
        mu_risk: float = 0.0,
        max_variance: Optional[float] = None,
        solver: Optional[str] = None,
        style_tol: float = 0.05,
    ) -> None:
        self.lambda_turnover   = lambda_turnover
        self.max_weight        = max_weight
        self.max_turnover      = max_turnover
        self.industry_tol      = industry_tol
        self.industry_tol_max  = industry_tol_max
        self.industry_tol_step = industry_tol_step
        self.mu_risk           = mu_risk
        self.max_variance      = max_variance
        self.solver            = solver
        self.style_tol         = style_tol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        alpha_t: np.ndarray,
        w_prev: np.ndarray,
        X_industry: np.ndarray,
        w_benchmark: np.ndarray,
        X_risk: Optional[np.ndarray] = None,
        F_half: Optional[np.ndarray] = None,
        delta_std: Optional[np.ndarray] = None,
        w_benchmark_stock: Optional[np.ndarray] = None,
        X_style: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[float], bool]:
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
        X_industry : np.ndarray, shape (n, K_ind)
            Binary industry dummy matrix.
        w_benchmark : np.ndarray, shape (K_ind,)
            Benchmark industry weight vector (sums to 1).
        X_risk : np.ndarray or None, shape (n, K)
            Factor exposure matrix from the risk model (style + industry factors).
            If all three risk inputs are provided, the risk term is activated.
        F_half : np.ndarray or None, shape (K, K)
            Upper Cholesky factor of the factor covariance matrix F_t.
            Satisfies: F = F_half.T @ F_half.
        delta_std : np.ndarray or None, shape (n,)
            Per-stock idiosyncratic standard deviation (sqrt(Δ_{ii})).
        w_benchmark_stock : np.ndarray or None, shape (n,)
            Stock-level benchmark weights (e.g. market-cap weighted).  When
            provided with X_style, adds |w_active' X_factor| <= style_tol constraints.
        X_style : np.ndarray or None, shape (n, K_style)
            Style factor exposure matrix.  Must be provided with w_benchmark_stock.

        Returns
        -------
        w_star : np.ndarray, shape (n,)
            Optimal portfolio weights.  Sum to 1, all non-negative.
        tol_used : float or None
            Industry tolerance actually used.  Equal to ``industry_tol`` when
            the problem solved on the first try; larger if relaxed; None if the
            industry constraint was dropped as a last resort.
        fallback_used : bool
            True if the solution was rejected due to constraint violation
            or solver failure and equal weights were returned instead.
        """
        n = len(alpha_t)

        # Handle edge cases
        if n == 0:
            return np.array([]), None, False
        if n == 1:
            return np.array([1.0]), self.industry_tol, False

        alpha_t  = np.asarray(alpha_t,  dtype=float)
        w_prev   = np.asarray(w_prev,   dtype=float)
        X_ind    = np.asarray(X_industry, dtype=float)
        w_bench  = np.asarray(w_benchmark, dtype=float)

        w = cp.Variable(n, nonneg=True)

        # Objective: maximise alpha return minus turnover-aversion penalty.
        # lambda_turnover controls the signal-vs-stability trade-off;
        # it is NOT a monetary cost rate (see class docstring).
        obj_expr = w @ alpha_t - self.lambda_turnover * 0.5 * cp.norm1(w - w_prev)

        # Optional risk penalty: -0.5 * mu_risk * w^T Σ w
        # Decomposed as: w^T Σ w = ||F_half @ (X_risk^T w)||² + ||delta_std ⊙ w||²
        # This avoids forming the full N×N covariance matrix and enables SOCP.
        risk_active = (
            X_risk is not None
            and F_half is not None
            and delta_std is not None
        )
        risk_quad = None
        if risk_active:
            X_r  = np.asarray(X_risk,    dtype=float)   # (n, K)
            F_h  = np.asarray(F_half,     dtype=float)   # (K, K)
            d_s  = np.asarray(delta_std,  dtype=float)   # (n,)

            # Factor portfolio exposures: z = F_half @ X_risk^T w,  shape (K,)
            z = F_h @ (X_r.T @ w)
            # Risk quadratic: ||z||² + ||delta_std ⊙ w||²
            risk_quad = cp.sum_squares(z) + cp.sum_squares(cp.multiply(d_s, w))

            if self.mu_risk > 0:
                obj_expr = obj_expr - 0.5 * self.mu_risk * risk_quad

        objective = cp.Maximize(obj_expr)

        # Base constraints (never relaxed)
        base_constraints = [
            cp.sum(w) == 1,
            w <= self.max_weight,
        ]
        if self.max_turnover is not None:
            # 0.5||w - w_prev||_1 <= max_turnover  <=>  ||w - w_prev||_1 <= 2*max_turnover
            base_constraints.append(cp.norm1(w - w_prev) <= 2 * self.max_turnover)

        if risk_active and self.max_variance is not None and risk_quad is not None:
            # w^T Σ w <= 2 * max_variance
            base_constraints.append(risk_quad <= 2 * self.max_variance)

        # Style factor neutrality: |w_active' X_factor_k| <= style_tol
        w_bench_stock = np.asarray(w_benchmark_stock, dtype=float) if w_benchmark_stock is not None else None
        X_style_arr = np.asarray(X_style, dtype=float) if X_style is not None else None
        style_active = (
            w_bench_stock is not None
            and X_style_arr is not None
            and X_style_arr.size > 0
            and len(w_bench_stock) == n
            and X_style_arr.shape[0] == n
        )
        if style_active:
            w_active = w - w_bench_stock
            for k in range(X_style_arr.shape[1]):
                x_k = X_style_arr[:, k]
                base_constraints.append(cp.sum(cp.multiply(w_active, x_k)) >= -self.style_tol)
                base_constraints.append(cp.sum(cp.multiply(w_active, x_k)) <= self.style_tol)

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
                ok, max_viol, msg = self._validate_solution(
                    problem, w.value, risk_active,
                    X_r if risk_active else None,
                    F_h if risk_active else None,
                    d_s if risk_active else None,
                    w_bench_stock if style_active else None,
                    X_style_arr if style_active else None,
                )
                if ok:
                    return np.clip(w.value, 0.0, None), float(tol), False
                # Constraint violation: try next tol

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
            ok, max_viol, msg = self._validate_solution(
                problem_no_ind, w.value, risk_active,
                X_r if risk_active else None,
                F_h if risk_active else None,
                d_s if risk_active else None,
                w_bench_stock if style_active else None,
                X_style_arr if style_active else None,
            )
            if ok:
                return np.clip(w.value, 0.0, None), None, False

        # Absolute fallback: equal weight (solver failed or constraints violated)
        warnings.warn(
            "Optimisation failed or solution violates constraints. "
            "Falling back to equal weights.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(n, 1.0 / n), None, True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_solution(
        self,
        problem: "cp.Problem",
        w_value: np.ndarray,
        risk_active: bool,
        X_risk: Optional[np.ndarray],
        F_half: Optional[np.ndarray],
        delta_std: Optional[np.ndarray],
        w_benchmark_stock: Optional[np.ndarray] = None,
        X_style: Optional[np.ndarray] = None,
    ) -> tuple[bool, float, str]:
        """Check that the solution satisfies all constraints within tolerance.

        Returns
        -------
        ok : bool
            True if all constraints satisfied.
        max_viol : float
            Maximum violation across constraints.
        msg : str
            Description for logging.
        """
        tol = 1e-5
        max_viol = 0.0
        worst_msg = ""

        # Check cvxpy constraint violations
        for c in problem.constraints:
            try:
                v = c.violation()
                if v is not None:
                    arr = np.atleast_1d(np.asarray(v))
                    m = float(np.abs(arr).max())
                    if m > max_viol:
                        max_viol = m
                        worst_msg = f"constraint violation {m:.2e}"
            except (ValueError, TypeError):
                pass

        # Explicit max_variance check (cvxpy canonicalization may obscure this)
        if (
            risk_active
            and self.max_variance is not None
            and X_risk is not None
            and F_half is not None
            and delta_std is not None
        ):
            w_val = np.asarray(w_value, dtype=float)
            z = F_half @ (X_risk.T @ w_val)
            actual_var = float(np.dot(z, z) + np.dot(delta_std * w_val, delta_std * w_val))
            limit = 2 * self.max_variance * (1 + 1e-5)
            if actual_var > limit:
                viol = actual_var - 2 * self.max_variance
                if viol > max_viol:
                    max_viol = viol
                worst_msg = (
                    f"max_variance violated: actual={actual_var:.2e} "
                    f"> limit={2*self.max_variance:.2e}"
                )

        # Explicit style factor neutrality check
        if (
            w_benchmark_stock is not None
            and X_style is not None
            and X_style.size > 0
        ):
            w_val = np.asarray(w_value, dtype=float)
            w_active = w_val - np.asarray(w_benchmark_stock, dtype=float)
            for k in range(X_style.shape[1]):
                exp_k = float(np.dot(w_active, X_style[:, k]))
                if abs(exp_k) > self.style_tol + 1e-5:
                    viol = abs(exp_k) - self.style_tol
                    if viol > max_viol:
                        max_viol = viol
                    worst_msg = (
                        f"style factor k={k} violated: |exposure|={abs(exp_k):.2e} "
                        f"> tol={self.style_tol:.2e}"
                    )

        ok = max_viol <= tol
        return ok, max_viol, worst_msg or "unknown"

    def _solve_problem(self, problem: "cp.Problem") -> None:
        """Invoke the cvxpy solver, suppressing verbose output."""
        solver_kwargs: dict = {"verbose": False}
        if self.solver is not None:
            solver_kwargs["solver"] = self.solver
        try:
            problem.solve(**solver_kwargs)
        except cp.SolverError as e:
            warnings.warn(
                f"Solver failed: {e}. Status may be invalid.",
                RuntimeWarning,
                stacklevel=2,
            )
