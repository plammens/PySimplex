# import argparse as arg
# import sys
import re
import numpy as np
import simplex

"""
argparser = arg.ArgumentParser()
argparser.add_argument("num", type=int, default=1)
argparser.add_argument("prob", type=int, default=1)
args = argparser.parse_args(sys.argv[1:])

num = args.num
prob = args.prob
"""

num = 41
prob = 1

with open("pm18_exercici_simplex_dades.txt", 'r') as file:
    def skip_to(patt: re.Pattern):
        while True:
            line = file.readline()
            if re.search(patt, line):
                return line

    skip_to(r"cjt. dades {}, problema PL {}".format(num, prob))

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

    skip_to("b=")
    b = parse_mat()
    pass

simplex.simplex(A, b, c)
