"""
This script takes three arguments (upon being called, from stdin): num, prob, rule.
It parses the data from the corresponding problem (problem set `num`,
problem number `prob`) and executes the simplex algorithm from the `simplex` module
with that data.
"""


import argparse as arg
import sys
import re
import numpy as np
import simplex

def prob_set_num_type(x):
    x = int(x)
    if x not in range(1, 80):
        raise arg.ArgumentError("Invalid problem set number (should be in [1-79])")
    return x

def prob_num_type(x):
    x = int(x)
    if x not in range(1, 5):
        raise arg.ArgumentError("Invalid problem number (should be in [1-4])")
    return x

argparser = arg.ArgumentParser()
argparser.add_argument("num", type=prob_set_num_type, default=1,
                       help="problem set number within 1-79")
argparser.add_argument("prob", type=prob_num_type, default=1,
                       help="problem number within 1-4")
argparser.add_argument("--rule", type=str, default="bland",
                       choices=["bland", "minrc"],
                       help="pivoting rule for simplex algorithm")
args = argparser.parse_args(sys.argv[1:])

num = args.num
prob = args.prob
rule = 0 if args.rule == "bland" else 1


with open("pm18_exercici_simplex_dades.txt", 'r') as file:
    def skip_to(patt: re.Pattern):
        while True:
            line = file.readline()
            if re.search(patt, line):
                return line

    skip_to(r"cjt. dades {:>2}, problema PL {:>}".format(num, prob))

    def parse_mat():
        mat = None

        line = file.readline()
        match = re.search(r"Columns (\d+) through (\d+)$", line)

        def read_block():
            nonlocal line, mat

            block = []
            while line != "\n":
                row = [int(c_j) for c_j in re.findall("-?\d+", line)]
                block.append(row)
                line = file.readline()

            block = np.array(block)

            if mat is not None:
                mat = np.append(mat, block, axis=1)
            else:
                mat = block

        if bool(match):
            for _ in (0, 1):
                line = skip_to(re.compile(r"^ +(-?\d+ *){2,22}"))  # Skip to line with digits
                read_block()
        else:
            read_block()

        return np.matrix(mat) if mat.shape[0] > 1 else mat[0]

    skip_to(r"c=")
    c = parse_mat()

    skip_to(r"A=")
    A = parse_mat()

    skip_to(r"b=")
    b = parse_mat()


print("Solving problem set {}, problem number {}, with {} rule..."
        .format(num, prob, "Bland's" if rule == 0 else "minimal reduced cost"), end="\n\n")

simplex.simplex(A, b, c, rule)

print("\n\n")
