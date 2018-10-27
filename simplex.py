import numpy as np
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type


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
        temp_product = c[B] * B_inv  # Store product for efficiency
        for q in N:  # Read in lexicographical index order
            r = c[q] - temp_product * A[:, q]
            if r < 0:
                break
        else:
            print("Unlimited problem.")
            return 0  # Found optimal solution

        """Feasible basic direction"""
        d_B = -B_inv * A[q]

        """Maximum step length"""
        d_neg = [(-x_b[j] / d_B[j], j) for j in B if d_B[j] < 0]

        if len(d_neg) == 0:
            d_N = np.array([1 if i == q else 0 for i in N])
            d = np.concatenate((d_B, d_N))
            print("Unlimited problem. Feasible ray: {0}".format(d))
            return 1  # Flag problem as unlimited

        buffer = min(d_neg, key=(lambda tup: tup[0]))
        theta, p = buffer[0], buffer[1]  # Get theta and index of exiting basic variable
