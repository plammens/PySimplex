import numpy as np
import enum
from typing import Optional


class SolveResult:
    class ExitCode(enum.Enum):
        OPTIMUM = "optimal solution found"
        UNBOUNDED = "unlimited problem"
        UNFEASIBLE = "unfeasible problem"

    exit: ExitCode
    iterations: int
    _solution: Optional[np.ndarray]
    _base: Optional[set]
    _optimal_cost: Optional[float]
    _direction: Optional[np.ndarray]

    def __init__(self, exit: ExitCode, iterations: int,
                 solution: np.ndarray = None, base: set = None, optimal_cost: float = None,
                 direction: np.ndarray = None):
        self.exit = exit
        self.iterations = iterations
        self._solution = solution
        self._base = base
        self._optimal_cost = optimal_cost
        self._direction = direction

    @property
    def solution(self):
        self._check_exit(SolveResult.ExitCode.OPTIMUM, SolveResult.ExitCode.UNBOUNDED)
        return self._solution

    @property
    def base(self):
        self._check_exit(SolveResult.ExitCode.OPTIMUM, SolveResult.ExitCode.UNBOUNDED)
        return self._base

    @property
    def optimal_cost(self):
        self._check_exit(SolveResult.ExitCode.OPTIMUM)
        return self._optimal_cost

    @property
    def direction(self):
        self._check_exit(SolveResult.ExitCode.UNBOUNDED)
        return self._solution

    def _check_exit(self, *expected_codes: ExitCode):
        if self.exit not in expected_codes:
            raise AttributeError("invalid request: {}".format(self.exit))
