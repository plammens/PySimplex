# import argparse as arg
# import sys
import re
import numpy as np

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
    def skip_to(patt: str):
        while True:
            if re.search(patt, file.readline()):
                break

    skip_to(r"cjt. dades {}, problema PL {}".format(num, prob))

    def parse_mat():
        mat = []

        line = file.readline()
        match = re.search(r"Columns (\d+) through (\d+)$", line)

        def read_block():
            nonlocal line
            while line != "\n":
                row = [int(c_j) for c_j in re.findall("-?\d+", line)]
                mat.append(row)
                line = file.readline()

        if bool(match):
            file.readline()  # Blank line
            read_block()
            for i in range(3):
                file.readline()
            read_block()
        else:
            read_block()

        return np.matrix(mat) if len(mat) > 1 else np.array(mat[0])

    skip_to(r"c=")
    c = parse_mat()

    skip_to(r"A=")
    A = parse_mat()

    skip_to("b=")
    b = parse_mat()

