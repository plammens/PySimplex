"""
Luis Sierra & Paolo Lammens -- FME-UPC
"""

import numpy as np
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type


def simplex(A: matrix, b: np.array, c: np.array):
    """
    Outer "wrapper" for executing the simplex method: phase I and phase II.
    """

    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    """Error-checking"""
    if b.shape != (m,):
        raise ValueError("Incompatible dimensions: c_j has shape {}, expected {}.".format(b.shape, (m,)))
    if c.shape != (n,):
        raise ValueError("Incompatible dimensions: c has shape {}, expected {}.".format(c.shape, (n,)))

    """Phase I setup"""
    A[[i for i in range(m) if b[i] < 0]] *= -1  # Change sign of constraints
    b = np.abs(b)  # Idem

    A_I = matrix(np.concatenate((A, np.identity(m)), axis=1))  # Phase I constraint matrix
    x_I = np.concatenate((np.zeros(n), b))  # Phase I variable vector
    c_I = np.concatenate((np.zeros(n), np.ones(m)))  # Phase I c_j vector
    basic_I = set(range(n, n + m))  # Phase I basic variable set

    """Phase I execution"""
    print("Executing phase I...")
    ext_I, x_init, basic_init, z_I, d, it_I = simplex_core(A_I, c_I, x_I, basic_I)
    # ^ Exit code, initial BFS & basis, z_I, d (not needed) and no of iterations
    print("Phase I terminated.")

    assert ext_I == 0
    if any(j not in range(n) for j in basic_init):
        raise NotImplementedError("Artificial variables in basis")

    if z_I > 0:
        print("Infeasible problem (z_I = {} > 0).".format(z_I))
        return 2, None, None

    x_init = x_init[:n]

    print("Found initial BFS at x = {}.\n".format(x_init))

    """Phase II"""
    print("Executing phase II...")
    ext, x, basic, z, d, it_II = simplex_core(A, c, x_init, basic_init)
    print("Phase II terminated.\n")

    if ext == 0:
        print("Found optimal solution at x = {}. Optimal c_j: {}.".format(x, z))
    elif ext == 1:
        print("Unlimited problem. Found feasible ray d = {} from x = {}.".format(d, x))

    print("{} iterations in phase I, {} iterations in phase II.".format(it_I, it_II))

    return ext, x, z, d


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

    it = 1
    while True:
        print("\tIteration no. {}:".format(it), end='')

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
            print("\tfound optimum at x = {0}".format(x))
            return 0, x, basic, z, None, it  # Found optimal solution

        """Feasible basic direction"""
        d = np.array([np.asscalar(-B_inv[B.index(j), :] * A[:, q]) if j in basic else 1 if j == q else 0
                      for j in range(n)])

        """Maximum step length"""
        neg = [(-x[i] / d[i], i) for i in basic if d[i] < 0]

        if len(neg) == 0:
            print("\tidentified unlimited problem")
            return 1, x, basic, None, d, it  # Flag problem as unlimited

        buffer = min(neg, key=(lambda tuple_: tuple_[0]))
        theta, p = buffer[0], buffer[1]  # Get theta and index of exiting basic variable

        """Variable updates"""
        x = x + theta * d  # Update all variables
        assert x[p] == 0

        z = z + theta * r_q

        basic = basic - {p} | {q}  # Update basis set
        nonbasic = nonbasic - {q} | {p}  # Update nonbasic set

        """Print status update"""
        print(
            "\tq = {:>2} \trq = {:>4} \tp = {:>2} \ttheta* = {:>5} \tz = {:>5}"
            .format(it, q, r_q, p, theta, z)
        )

        it += 1


if __name__ == '__main__':
    A_ = matrix([[2, 1, 1, 0], [0, 1, 0, 1]])
    b_ = np.array([8, 6])
    c_ = np.array([1, 1, 0, 0])

    simplex(A_, b_, c_)
