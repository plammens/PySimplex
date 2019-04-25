from pysimplex.solveresult import SolveResult


def print_boxed(msg: str) -> None:
    """
    Utility for printing pretty boxes.
    :param msg: message to be printed
    """

    lines = msg.splitlines()
    max_len = max(len(line) for line in lines)

    if max_len > 100:
        raise ValueError("Overfull box")

    print('-' * (max_len + 4))
    for line in lines:
        print('| ' + line + ' ' * (max_len - len(line)) + ' |')
    print('-' * (max_len + 4))
