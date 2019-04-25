"""
~Mathematical Programming~
Simplex implementation assignment.

Luis Sierra & Paolo Lammens -- GM FME-UPC
luis.sierra.muntane@estudiant.upc.edu
paolo.matias.lammens@estudiant.upc.edu
"""

from pysimplex.core import *
from pysimplex.utils import *



def simplex(costs: np.ndarray, constraints: np.ndarray, independent_terms: np.ndarray,
            *, rule: PivotingRule = PivotingRule.BLAND,  verbose: bool = True) -> SolveResult:
    """
    Outer wrapper for executing the simplex method, phase I and phase II.
    :param costs:  cost coefficients for each variable
    :param constraints: matrix A of the standard-form constraints Ax = b
    :param independent_terms: vector b of the standard-form constraints Ax = b
    :param rule: variable selection rule (e.g. Bland's)
    :param verbose: if True, prints the outcome of each step to stdout.
    """
    lp = LinearProgrammingProblem(costs, constraints, independent_terms)
    return Simplex(lp).solve(rule=rule, verbose=verbose)



class Simplex:
    lp: LinearProgrammingProblem

    def __init__(self, lp: LinearProgrammingProblem):
        self.lp = lp

    def solve(self, rule: PivotingRule = PivotingRule.BLAND, *, verbose: bool = True) \
            -> SolveResult:
        """
        Outer "wrapper" for executing the simplex method: phase I and phase II.
        :param rule: variable selection rule (e.g. Bland's)
        :param verbose: if True, prints the outcome of each step to stdout.
        """
        if verbose:
            np.set_printoptions(precision=3, threshold=10, edgeitems=4, linewidth=120)

        phaseI_result = self.run_phaseI(verbose=verbose)
        if phaseI_result.optimal_cost > 0:
            if verbose: _print_unfeasible(phaseI_result)
            return SolveResult(SolveResult.ExitCode.UNFEASIBLE, iterations=phaseI_result.iterations)

        phaseII_result = self.run_phaseII(phaseI_result, verbose=verbose)
        if verbose:
            _print_phaseII(phaseII_result)
            _print_iterations(phaseI_result, phaseII_result)
        return phaseII_result


    def run_phaseI(self, *, verbose: bool) -> SolveResult:
        m, n = self.lp.nrows, self.lp.ncols

        mask = self.lp.independent_terms < 0
        self.lp.constraints[mask] *= -1  # Change sign of constraints
        self.lp.independent_terms[mask] *= -1  # Idem

        A_I = np.concatenate((self.lp.constraints, np.identity(m)), axis=1)
        c_I = np.concatenate((np.zeros(n), np.ones(m)))

        phaseI = LinearProgrammingProblem(costs=c_I, constraints=A_I,
                                          independent_terms=self.lp.independent_terms)
        x_I = np.concatenate((np.zeros(n), self.lp.independent_terms))
        basic_I = set(range(n, n + m))

        if verbose: print("Executing phase I...")
        result = SimplexCore(phaseI, initial_bfs=x_I, basic_indices=basic_I).solve(verbose=verbose)
        if verbose: print("Phase I terminated.")
        assert result.exit is SolveResult.ExitCode.OPTIMUM

        return result

    def run_phaseII(self, phaseI_result: SolveResult, *, verbose: bool) -> SolveResult:
        # Get initial BFS for original problem (without artificial variables):
        self._remove_artificial_variables(phaseI_result)
        init_bfs = phaseI_result.solution[:self.lp.ncols]
        init_base = phaseI_result.base
        if verbose: print("Found initial BFS at x = \n{}.\n".format(init_bfs))

        if verbose: print("Executing phase II...")
        result = SimplexCore(self.lp, init_bfs, init_base).solve(verbose=verbose)
        if verbose: print("Phase II terminated.\n")

        return result


    def _remove_artificial_variables(self, phaseI: SolveResult):
        if any(j not in range(self.lp.ncols) for j in phaseI.base):
            # If some artificial variable is in the basis for the initial BFS, exit:
            raise NotImplementedError("Artificial variables in basis")



def _print_unfeasible(phaseI: SolveResult):
    print("\n")
    print_boxed("Unfeasible problem (z_I = {:.6g} > 0).".format(phaseI.optimal_cost))
    print("{} iterations in phase I.".format(phaseI.iterations), end='\n\n')


def _print_phaseII(phaseII: SolveResult):
    n = len(phaseII.solution)

    if phaseII.exit is SolveResult.ExitCode.OPTIMUM:
        print_boxed("Found optimal solution at x =\n{}.\n\n".format(phaseII.solution) +
                    "Basic indices: {}\n".format(phaseII.base) +
                    "Nonbasic indices: {}\n\n".format(set(range(n)) - phaseII.base) +
                    "Optimal cost: {}.".format(phaseII.optimal_cost))

    elif phaseII.exit is SolveResult.ExitCode.UNBOUNDED:
        print_boxed("Unbounded problem. Found feasible ray "
                    "d =\n{}\nfrom x =\n{}.".format(phaseII.direction, phaseII.solution))

    else: assert False


def _print_iterations(phaseI: SolveResult, phaseII: SolveResult):
    print("{} iterations in phase I, {} iterations in phase II "
          "({} total).".format(phaseI.iterations, phaseII.iterations,
                               phaseI.iterations + phaseII.iterations),
          end='\n\n')
