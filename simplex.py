"""
Luis Sierra & Paolo Lammens -- FME-UPC
"""

import numpy as np
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type


def simplex(A: matrix, b: np.array, c: np.array) -> (int, np.array, float):
    """
    Outer "wrapper" for executing the simplex method: phase I and phase II.
    """

    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    """Error-checking"""
    if b.shape != (m,):
        raise ValueError("Incompatible dimensions: b has shape {}, expected {}.".format(b.shape, (m,)))
    if c.shape != (n,):
        raise ValueError("Incompatible dimensions: c has shape {}, expected {}.".format(c.shape, (n,)))

    """Phase I setup"""
    A[[i for i in range(m) if b[i] < 0]] *= -1  # Change sign of constraints
    b = np.abs(b)  # Idem

    A_I = np.concatenate((A, np.identity(m)), axis=1)  # Phase I constraint matrix
    x_I = np.concatenate((np.zeros(n), b))  # Phase I variable vector
    c_I = np.concatenate((np.zeros(n), np.ones(m)))  # Phase I cost vector
    base_I = set(range(n, n + m))  # Phase I basic variable set

    """Phase I execution"""
    print("Executing phase I...")
    ext_I, x_init, basic_init, z_I, d = simplex_core(A_I, c_I, x_I, base_I)
    # ^ Exit code, initial BFS & basis, and z_I
    assert ext_I == 0

    x_init = x_init[:n]

    if z_I > 0:
        print("Infeasible problem")
        return 2, None, None

    print()

    """Phase II"""
    print("Executing phase II...")
    ext, x, basic, z, d = simplex_core(A, c, x_init, basic_init)

    if ext == 0:
        print("Found optimal solution at x = {}. Optimal cost: {}.".format(x, z))
        return 0, x, z
    elif ext == 1:
        print("Unlimited problem. Found feasible ray d = {} from x = {}.".format(d, x))
        return 1, x, None, d


def simplex_core(A: matrix, c: np.array, x: np.array, basic: set) -> (int, np.array, set, float, np.array):
    """
    This function executes the simplex algorithm iteratively until it
    terminates. It is the core function of this project.
    """

    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    assert c.shape == (n,) and x.shape == (n,)
    assert len(basic) == m and \
        all(i in range(n) for i in basic)  # Make sure that basic is a valid base

    nonbasic = set(range(n)) - basic  # Nonbasic index set

    z = np.dot(c, x)

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
            return 0, x, basic, z, None  # Found optimal solution

        """Feasible basic direction"""
        d = np.array([np.asscalar(-B_inv[B.index(j), :] * A[:, q]) if j in basic else 1 if j == q else 0
                      for j in range(n)])

        """Maximum step length"""
        neg = [(-x[i] / d[i], i) for i in basic if d[i] < 0]

        if len(neg) == 0:
            print("Unlimited problem. Feasible ray: {0}".format(d))
            return 1, x, basic, None, d  # Flag problem as unlimited

        buffer = min(neg, key=(lambda tuple_: tuple_[0]))
        theta, p = buffer[0], buffer[1]  # Get theta and index of exiting basic variable

        """Variable updates"""
        x = x + theta * d  # Update all variables
        assert x[p] == 0

        z = z + theta * r_q

        basic = basic - {p} | {q}  # Update basis set
        nonbasic = nonbasic - {q} | {p}  # Update nonbasic set


if __name__ == '__main__':
    A_ = matrix([[2, 1, 1, 0], [0, 1, 0, 1]])
    b_ = np.array([8, 6])
    c_ = np.array([1, 1, 0, 0])

    simplex(A_, b_, c_)
