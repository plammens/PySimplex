import numpy as np
from numpy.matlib import matrix


class LinearProgrammingProblem:
    costs: np.ndarray
    constraints: matrix
    independent_terms: np.ndarray

    def __init__(self, costs: np.array, constraints: matrix, independednt_terms: np.array):
        self.constraints = constraints
        if costs.shape != (self.cols) or independednt_terms.shape != (self.nrows):
            raise ValueError("shapes do not match")
        self.costs, self.independent_terms = costs, independednt_terms

    @property
    def nrows(self):
        return self.constraints.shape[0]

    @property
    def ncols(self):
        return self.constraints.shape[1]
