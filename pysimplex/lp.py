import numpy as np
import sympy


class LinearProgrammingProblem:
    costs: np.ndarray
    constraints: np.ndarray
    independent_terms: np.ndarray

    def __init__(self, costs: np.array, constraints: np.ndarray, independent_terms: np.array):
        if costs.ndim != 1: raise ValueError("costs should be a 1D vector")
        if constraints.ndim != 2: raise ValueError("constraints should be a 2D matrix")
        if independent_terms.ndim != 1: raise ValueError("independent terms should be a 1D vector")

        self.constraints = constraints
        self._remove_ld_rows()
        if costs.shape[0] != self.ncols or independent_terms.shape[0] != self.nrows:
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
