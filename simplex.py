"""
Luis Sierra & Paolo Lammens -- FME-UPC
"""

import numpy as np
from numpy.linalg import inv  # Matrix inverse
from numpy.matlib import matrix  # Matrix data type

np.set_printoptions(precision=3, threshold=10, edgeitems=4, linewidth=120)  # Prettier array printing

epsilon = 10**(-10)  # Global truncation threshold


def simplex(A: matrix, b: np.array, c: np.array, rule: int):
    """
    Outer "wrapper" for executing the simplex method: phase I and phase II.
    """

    m, n = A.shape[0], A.shape[1]  # no. of rows, columns of A, respectively

    """Error-checking"""
    if n < m:
        raise ValueError("Incompatible dimensions "
                         "(no. of variables : {} > {} : no.of constraints".format(n, m))
    if b.shape != (m,):
        raise ValueError("Incompatible dimensions: c_j has shape {}, expected {}.".format(b.shape, (m,)))
    if c.shape != (n,):
        raise ValueError("Incompatible dimensions: c has shape {}, expected {}.".format(c.shape, (n,)))


    "Check full rank matrix"
    if not np.linalg.matrix_rank(A) == m:
        # Remove ld rows:
        A = A[[i for i in range(m) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(n))], :]
        m = A.shape[0]  # Update no. of rows


    """Phase I setup"""
    A[[i for i in range(m) if b[i] < 0]] *= -1  # Change sign of constraints
    b = np.abs(b)  # Idem

    A_I = matrix(np.concatenate((A, np.identity(m)), axis=1))  # Phase I constraint matrix
    x_I = np.concatenate((np.zeros(n), b))  # Phase I variable vector
    c_I = np.concatenate((np.zeros(n), np.ones(m)))  # Phase I c_j vector
    basic_I = set(range(n, n + m))  # Phase I basic variable set


    """Phase I execution"""
    print("Executing phase I...")
    ext_I, x_init, basic_init, z_I, d, it_I = simplex_core(A_I, c_I, x_I, basic_I, rule)
    # ^ Exit code, initial BFS & basis, z_I, d (not needed) and no. of iterations
    print("Phase I terminated.")

    assert ext_I == 0  # assert that phase I has an optimal solution (and is not unlimited)
    if z_I > 0:
        print("\n")
        print_boxed("Unfeasible problem (z_I = {:.6g} > 0).".format(z_I))
        print("{} iterations in phase I.".format(it_I), end='\n\n')
        return 2, None, None
    if any(j not in range(n) for j in basic_init):
        # If some artificial variable is in the basis for the initial BFS, exit:
        raise NotImplementedError("Artificial variables in basis")

    x_init = x_init[:n]

    print("Found initial BFS at x = \n{}.\n".format(x_init))


    """Phase II execution"""
    print("Executing phase II...")
    ext, x, basic, z, d, it_II = simplex_core(A, c, x_init, basic_init, rule)
    print("Phase II terminated.\n")

    if ext == 0:
        print_boxed("Found optimal solution at x =\n{}.\n\n".format(x) +
                    "Basic indexes: {}\n".format(basic) +
                    "Nonbasic indexes: {}\n\n".format(set(range(n)) - basic) +
                    "Optimal cost: {}.".format(z))
    elif ext == 1:
        print_boxed("Unlimited problem. Found feasible ray d =\n{}\nfrom x =\n{}.".format(d, x))

    print("{} iterations in phase I, {} iterations in phase II ({} total).".format(it_I, it_II, it_I + it_II),
          end='\n\n')

    return ext, x, z, d


def simplex_core(A: matrix, c: np.array, x: np.array, basic: set, rule: int) \
        -> (int, np.array, set, float, np.array):
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
        r_q, q  = 0, 0  # Initialize reduced cost and entering var. index
        temp_product = c[B] * B_inv  # Store product for efficiency

        if rule == 0:
            optimum = True

            for q in N:  # Read in lexicographical index order
                r_q = np.asscalar(c[q] - temp_product * A[:, q])
                if r_q < 0:
                    optimum = False
                    break

        elif rule == 1:
            optimum = False

            r_q, q = min([(np.asscalar(c[q] - temp_product * A[:, q]), q) for q in N], key=(lambda tup: tup[0]))
            if r_q >= 0:
                optimum = True
        else:
            raise ValueError("Invalid rule for variable selection")

        if optimum:
            print("\tfound optimum")
            return 0, x, basic, z, None, it  # Found optimal solution


        """Feasible basic direction"""
        d = np.array([trunc(np.asscalar(-B_inv[B.index(j), :] * A[:, q]))
                      if j in basic else 1 if j == q else 0
                      for j in range(n)])


        """Maximum step length"""
        neg = [(-x[i] / d[i], i) for i in basic if d[i] < 0]

        if len(neg) == 0:
            print("\tidentified unlimited problem")
            return 1, x, basic,  None, d, it  # Flag problem as unlimited and return ray

        buffer = min(neg, key=(lambda tuple_: tuple_[0]))
        theta, p = buffer[0], buffer[1]  # Get theta and index of exiting basic variable


        """Variable updates"""
        x = np.array([trunc(var) for var in (x + theta * d)])  # Update all variables
        assert x[p] == 0

        z = trunc(z + theta * r_q)  # Update obj. function value

        basic = basic - {p} | {q}  # Update basis set
        nonbasic = nonbasic - {q} | {p}  # Update nonbasic set


        """Print status update"""
        print(
            "\tq = {:>2} \trq = {:>9.2f} \tp = {:>2d} \ttheta* = {:>5.4f} \tz = {:<9.2f}"
                .format(q + 1, r_q, p + 1, theta, z)
        )

        it += 1


def print_boxed(msg: str) -> None:
    """Utility for printing pretty boxes."""

    lines = msg.splitlines()
    max_len = max(len(line) for line in lines)

    if max_len > 100:
        raise ValueError("Overfull box")

    print('-' * (max_len + 4))
    for line in lines:
        print('| ' + line + ' ' * (max_len - len(line)) + ' |')
    print('-' * (max_len + 4))


def trunc(x: float) -> float:
    return x if abs(x) >= epsilon else 0