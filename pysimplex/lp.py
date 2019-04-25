import numpy as np
import sympy


class LinearProgrammingProblem:
    costs: np.ndarray
    constraints: np.ndarray
    independent_terms: np.ndarray

    def __init__(self, costs: np.array, constraints: np.ndarray, independent_terms: np.array):
        self.constraints = constraints
        self._remove_ld_rows()
        if costs.shape != self.ncols or independent_terms.shape != self.nrows:
            raise ValueError("shapes do not match")
        self.costs, self.independent_terms = costs, independent_terms

    @property
    def nrows(self):
        return self.constraints.shape[0]

    @property
    def ncols(self):
        return self.constraints.shape[1]

    def _remove_ld_rows(self):
        if not np.linalg.matrix_rank(self.constraints) == self.nrows:
            # Remove ld rows:
            _, li_indexes = sympy.Matrix(self.constraints).T.rref()
            self.constraints = self.constraints[li_indexes, :]
