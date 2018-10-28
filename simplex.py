import numpy as np
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type


def wrapper(A: matrix, B: set):
    pass


def simplex(A: matrix, c: np.array, x: np.array, basic: set) -> int:
    """
    This function executes the simplex algorithm iteratively until it
    terminates. It is the core function of this project, and is intended for
    internal use only.
    """

    z = np.dot(c, x)
    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    assert len(c) == n and len(x) == n
    assert len(basic) == m and \
           all(i in range(n) for i in basic)  # Make sure that basic is a valid base

    nonbasic = set(range(n)) - basic  # Nonbasic index set

    while True:
        B, N = list(basic), list(nonbasic)  # Convert to list (from set) to simplify use as indexing expr.

        B_mat = A[:, B]  # Get basic matrix (all rows of A, columns specified by basic)

        # Calculate inverse:
        B_inv = inv(B_mat)

        """Optimality test"""
        temp_product = c[B] * B_inv  # Store product for efficiency
        for q in N:  # Read in lexicographical index order
            r_q = np.asscalar(c[q] - temp_product * A[:, q])
            if r_q < 0:
                break
        else:
            print("Found optimal solution at x = {0}".format(x))
            return 0  # Found optimal solution

        """Feasible basic direction"""
        d = np.array([np.asscalar(-B_inv[B.index(j), :] * A[:, q]) if j in basic else 1 if j == q else 0
                      for j in range(n)])

        """Maximum step length"""
        theta = None  # Sentinel value
        for i in [i for i in basic if d[i] < 0]:
            candidate = -x[i] / d[i]
            if theta is None or candidate < theta:
                theta = candidate
                p = i

        if theta is None:
            print("Unlimited problem. Feasible ray: {0}".format(d))
            return 1  # Flag problem as unlimited

        """Variable updates"""
        x = x + theta * d  # Update all variables
        assert x[p] == 0

        z = z + theta * r_q

        basic = basic - {p} | {q}  # Update basis set
        nonbasic = nonbasic - {q} | {p}  # Update nonbasic set


if __name__ == '__main__':
    A = matrix([[2, 1, 1, 0], [0, 1, 0, 1]])
    c = np.array([1, 1, 0, 0])
    x = np.array([0, 6, 2, 0])
    basis = {1, 2}
    simplex(A, c, x, basis)
