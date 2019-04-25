"""
~Mathematical Programming~
Simplex implementation assignment.

Luis Sierra & Paolo Lammens -- GM FME-UPC
luis.sierra.muntane@estudiant.upc.edu
paolo.matias.lammens@estudiant.upc.edu
"""

import numpy as np
from numpy.matlib import matrix  # Matrix data type

from pysimplex.utils import *

np.set_printoptions(precision=3, threshold=10, edgeitems=4, linewidth=120)  # Prettier array printing


def simplex(A: matrix, b: np.array, c: np.array, rule: int = 0) -> (int, np.array, float, np.array):
    """
    Outer "wrapper" for executing the simplex method: phase I and phase II.

    :param A: constraint matrix
    :param b: independent terms in constraints
    :param c: costs vector
    :param rule: variable selection rule (e.g. Bland's)

    This function prints the outcome of each step to stdout.
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


    """Check full rank matrix"""
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
    ext_I, x_init, basic_init, z_I, _, it_I = simplex_core(A_I, c_I, x_I, basic_I, rule)
    # ^ Exit code, initial BFS, basis, z_I, d (not needed) and no. of iterations
    print("Phase I terminated.")

    assert ext_I == 0  # assert that phase I has an optimal solution (and is not unbounded)
    if z_I > 0:
        print("\n")
        print_boxed("Unfeasible problem (z_I = {:.6g} > 0).".format(z_I))
        print("{} iterations in phase I.".format(it_I), end='\n\n')
        return 2, None, None, None
    if any(j not in range(n) for j in basic_init):
        # If some artificial variable is in the basis for the initial BFS, exit:
        raise NotImplementedError("Artificial variables in basis")

    x_init = x_init[:n]  # Get initial BFS for original problem (without artificial vars.)

    print("Found initial BFS at x = \n{}.\n".format(x_init))


    """Phase II execution"""
    print("Executing phase II...")
    ext, x, basic, z, d, it_II = simplex_core(A, c, x_init, basic_init, rule)
    print("Phase II terminated.\n")

    if ext == 0:
        print_boxed("Found optimal solution at x =\n{}.\n\n".format(x) +
                    "Basic indices: {}\n".format(basic) +
                    "Nonbasic indices: {}\n\n".format(set(range(n)) - basic) +
                    "Optimal cost: {}.".format(z))
    elif ext == 1:
        print_boxed("Unbounded problem. Found feasible ray d =\n{}\nfrom x =\n{}.".format(d, x))

    print("{} iterations in phase I, {} iterations in phase II ({} total).".format(it_I, it_II, it_I + it_II),
          end='\n\n')

    return ext, x, z, d
