from typing import Optional
from operator import itemgetter

from pysimplex.lp import *
from pysimplex.pivoting import PivotingRule
from pysimplex.solveresult import SolveResult


EPSILON = 10**(-10)  # Global truncation threshold
MAX_ITERATIONS = 10**5


class SimplexCore:
    lp: LinearProgrammingProblem
    basic_feasible_solution: np.ndarray
    basic_indices: list
    nonbasic_indices: set
    basic_matrix_inverse: np.ndarray  # inverse of basic matrix
    cost: float  # value of cost function

    def __init__(self, problem: LinearProgrammingProblem,
                 initial_bfs: np.array, basic_indices: set):
        self.lp = problem

        self._assert_is_valid_base(basic_indices)

        self.basic_feasible_solution = initial_bfs
        self.basic_indices = list(basic_indices)
        self.nonbasic_indices = set(range(self.lp.ncols)) - basic_indices
        self.basic_matrix_inverse = np.linalg.inv(self.lp.constraints[:, self.basic_indices])
        # noinspection PyTypeChecker
        self.cost = np.dot(self.lp.costs, self.basic_feasible_solution)  # Value of obj. function

    @property
    def x(self): return self.basic_feasible_solution

    @x.setter
    def x(self, value): self.basic_feasible_solution = value

    def solve(self, *, rule: PivotingRule = PivotingRule.BLAND,
              verbose: bool = True) -> SolveResult:
        """
        Applies the primal simplex algorithm to the problem stored in `self`.

        :param rule: pivoting rule
        :param verbose: whether to print status updates along the way
        :return: the result of applying the simplex algorithm with the initial
        basic feasible solution stored in self. If an optimal solution is found,
        `result.exit` will be SolveResult.ExitCode.OPTIMAL. If, otherwise, the
        problem is unbounded, it will be SolveResult.ExitCode.UNBOUNDED
        """

        # Some aliases for the conventional mathematical notation
        B, N = self.basic_indices, self.nonbasic_indices

        iteration: int = 1
        while iteration <= MAX_ITERATIONS:
            if verbose: print("\tIteration no. {}:".format(iteration), end='')

            is_optimal, q, r_q = self.optimality_test(rule=rule)
            if is_optimal: return self._return_optimal(iteration, verbose=verbose)

            d = self.get_basic_feasible_direction(entering_index=q)
            p, theta = self.get_maximum_step_length(direction=d)
            if p is None: return self._return_unbounded(direction=d, iteration=iteration,
                                                        verbose=verbose)

            self.move_along_direction(d, theta, p, q, r_q)

            self._print_status_update(q, r_q, B[p], theta)
            iteration += 1

        # If loop goes over max iterations (500):
        raise TimeoutError("Iterations maxed out (probably due to an endless loop)")


    def optimality_test(self, rule: PivotingRule) -> (bool, Optional[int], Optional[float]):
        """
        Optimality test. Tests if the current BFS (stored in self) is optimal.
        :return: a tuple containing whether the current solution is optimal, and if not,
        the entering nonbasic index (according to the given pivoting rule), and its
        associated reduced cost.
        """
        A, c = self.lp.constraints, self.lp.costs

        # Shadow prices:
        prices = self.lp.costs[self.basic_indices] @ self.basic_matrix_inverse

        # qth reduced cost:
        def r(q: int): return (c[q] - prices @ A[:, q]).item()

        if rule is PivotingRule.BLAND:  # Bland rule
            q, reduced_cost = min((tup for tup in ((q, r(q)) for q in self.nonbasic_indices)
                                   if tup[1] < 0), default=(None, None))
            return q is None, q, reduced_cost

        elif rule is PivotingRule.MIN_REDUCED_COST:  # Minimal reduced cost rule
            reduced_cost, q = min((r(q), q) for q in self.nonbasic_indices)
            return reduced_cost >= 0, q, reduced_cost

        else:
            raise ValueError("Invalid pivoting rule")


    def get_basic_feasible_direction(self, entering_index: int) -> np.ndarray:
        """
        Computes the basic feasible direction associated to the given
        nonbasic index.
        :param entering_index: the nonbasic index entering the base
        :return: the associated basic feasible direction
        """
        A, B_inv, B = self.lp.constraints, self.basic_matrix_inverse, self.basic_indices

        direction = np.zeros(self.lp.ncols)
        for i in range(self.lp.nrows):
            direction[B[i]] = trunc(-B_inv[i, :] @ A[:, entering_index])
        direction[entering_index] = 1

        return direction


    def get_maximum_step_length(self, direction: np.ndarray) -> (Optional[int], Optional[float]):
        """
        Computes the maximum step length given a basic feasible direction
        :param direction: basic feasible direction
        :return: a tuple with: the index `p` such that `B[p]` is the exiting basic
        index, and the maximum length of a step in the given direction
        """
        x, B, d = self.basic_feasible_solution, self.basic_indices, direction

        candidates = ((-x[B[i]] / d[B[i]], i) for i in range(self.lp.nrows) if d[B[i]] < 0)
        # noinspection PyTypeChecker
        theta, p = min(candidates, default=(None, None))
        return p, theta


    def move_along_direction(self, direction: np.ndarray, step_length: float,
                             p: int, q: int, r_q: float):
        x, d = self.x, direction
        B, N = self.basic_indices, self.nonbasic_indices
        B_inv = self.basic_matrix_inverse

        self.x += step_length*direction
        self.x = np.fromiter((trunc(elem) for elem in x), dtype=self.x.dtype, count=len(x))
        assert self.x[B[p]] == 0  # Update all variables
        self.cost = trunc(self.cost + step_length * r_q)  # Update obj. function value

        # Update inverse:
        for i in set(range(self.lp.nrows)) - {p}:
            B_inv[i, :] -= d[B[i]] / d[B[p]] * B_inv[p, :]
        B_inv[p, :] /= -d[B[p]]

        N.discard(q); N.add(B[p])  # Update nonbasic index set
        B[p] = q  # Update basic index list


    def _assert_is_valid_base(self, base: set):
        assert isinstance(base, set) and len(base) == self.lp.nrows and \
               all(i in range(self.lp.ncols) for i in base)


    def _return_optimal(self, iteration: int, *, verbose: bool) -> SolveResult:
        if verbose: print("\tfound optimum")
        return SolveResult(SolveResult.ExitCode.OPTIMUM, iterations=iteration,
                           solution=self.basic_feasible_solution,
                           base=set(self.basic_indices),
                           optimal_cost=self.cost)


    def _return_unbounded(self, direction: np.ndarray, iteration: int, *,
                          verbose: bool) -> SolveResult:
        if verbose: print("\tidentified unbounded problem")
        return SolveResult(SolveResult.ExitCode.UNBOUNDED, iterations=iteration,
                           solution=self.basic_feasible_solution,
                           direction=direction,
                           base=set(self.basic_indices))

    def _print_status_update(self,
                             entering_index: int,
                             entering_reduced_cost: float,
                             exiting_index: int,
                             step_length: float):
        print(
            "\tq = {:>2} \trq = {:>9.2f} \tB[p] = {:>2d} "
            "\ttheta* = {:>5.4f} \tz = {:<9.2f}".format(entering_index + 1, entering_reduced_cost,
                                                        exiting_index + 1, step_length,
                                                        self.cost)
        )


def trunc(x: float) -> float:
    """
    Returns 0 if x is smaller (in absolute value) than a EPSILON.
    """
    return x if abs(x) >= EPSILON else 0
