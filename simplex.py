import numpy as np
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type


def wrapper(A: matrix, B: set):
    pass


def simplex(A: matrix, c: np.array, basic: set, x: np.array, z: float) -> int:
    """
    This function executes the simplex algorithm iteratively until it
    terminates. It is the core function of this project, and is intended for
    internal use only.
    """
    
    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    assert len(basic) == m and \
           all(i in range(n) for i in basic)  # Make sure that basic is a valid base

    nonbasic = set(range(n)) - basic  # Nonbasic index set
    B_mat = A[:, list(basic)]  # Get basic matrix (all rows of A, columns specified by basic)

    while True:
        B, N = list(basic), list(nonbasic)  # Convert to list (from set) to simplify use as indexing expr.

        # Calculate inverse:
        B_inv = inv(B_mat)

        """Optimality test"""
        temp_product = c[B] * B_inv  # Store product for efficiency
        for q in N:  # Read in lexicographical index order
            r_q = c[q] - temp_product * A[:, q]
            if r_q < 0:
                break
        else:
            print("Found optimal solution at x = {0}".format(x))
            return 0  # Found optimal solution

        """Feasible basic direction"""
        d = [(-B_inv[j, :] * A[:, q]) if j in basic else 1 if j == q else 0 for j in range(n)]

        """Maximum step length"""
        d_neg = [(-x[i] / d[i], i) for i in basic if d[i] < 0]

        if len(d_neg) == 0:
            print("Unlimited problem. Feasible ray: {0}".format(d))
            return 1  # Flag problem as unlimited

        buffer = min(d_neg, key=(lambda tup: tup[0]))
        theta, p = buffer[0], buffer[1]  # Get theta and index of exiting basic variable

        """Variable updates"""
        basic = basic - {B[p]} | {q}  # Update basis set
        nonbasic = nonbasic - {q} | {B[p]}  # Update nonbasic set

        x = x + theta * d  # Update all basic variables
        assert x[p] == 0

        z = z + theta * r_q
