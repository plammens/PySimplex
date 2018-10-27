from numpy import *
from numpy.linalg import inv  # Matrix inverse


def wrapper(A: matrix, B: set):
    pass


def simplex(A: matrix, c: matrix, basic: set, x_b: matrix, z: float):
    """
    This function executes the simplex algorithm iteratively until it
    terminates. It is the core function of this project, and is intended for
    internal use only.
    """

    assert c.shape[1] == 1 and x_b.shape[1] == 1  # Make sure that c and x are (col.) vectors
    assert len(basic) == A.shape[0] and \
           all(i in range(A.shape[1]) for i in basic)  # Make sure that basic is a valid base

    nonbasic = set(range(A.shape[1])) - basic  # Nonbasic index set
    B_mat = A[:, list(basic)]  # Get basic matrix (all rows of A, columns specified by basic)

    while True:
        B, N = list(basic), list(nonbasic)  # Convert to list (from set) to simplify use as indexing expr.

        # Calculate inverse:
        B_inv = inv(B_mat)

        """Optimality test"""
        p = c[B] * B_inv  # Store product for efficiency
        for i in N:
            r = c[i] - p * A[:, i]
            if r < 0:
                break
        else:
            print("Unlimited problem.")
            return 1  # Flag the problem as unlimited problem
