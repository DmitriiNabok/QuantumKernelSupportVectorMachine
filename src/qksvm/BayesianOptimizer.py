# This code is part of Qiskit.
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiskit.algorithms.optimizers import Optimizer, OptimizerSupportLevel, OptimizerResult
from qiskit.algorithms.optimizers.optimizer import POINT


class BayesianOptimizer(Optimizer):
    """Bayesian optimization with skopt.

    https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html
    """

    def __init__(
        self,
        n_calls: int = 100,
        n_initial_points: int = 5,
        random_state: int = None,
        
    ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.

        Raises:
            MissingOptionalLibraryError: scikit-quant not installed
        """
        super().__init__()
        self._n_calls = n_calls
        self._n_initial_points = n_initial_points
        self._random_state = random_state

    def get_support_level(self):
        """Returns support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.ignored,
        }


    @property
    def settings(self) -> dict[str, Any]:
        return {
            "n_calls": self._n_calls,
            "n_initial_points": self._n_initial_points,
            "random_state": self._random_state,
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        
        from skopt import gp_minimize, forest_minimize
        from numpy import pi
        
        bounds = [(0, 2*pi) for _ in range(len(x0))]
        
        res = gp_minimize(
            fun,                                      # the function to minimize
            bounds,                                   # the bounds on each dimension of x
            acq_func="gp_hedge",                      # the acquisition function
            n_calls=self._n_calls,                    # the number of evaluations of f
            n_initial_points=self._n_initial_points,  # the number of random initialization points
            random_state=self._random_state,          # the random seed
        )

        optimizer_result = OptimizerResult()
        optimizer_result.x = res.x
        optimizer_result.fun = res.fun
        optimizer_result.nfev = len(res.x_iters)
        
        return optimizer_result