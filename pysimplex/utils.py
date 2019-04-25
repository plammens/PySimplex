

def print_status_update(entering_index: int,
                        entering_reduced_cost: float,
                        exiting_index: int,
                        step_length: float,
                        updated_cost: float):
    print(
        "\tq = {:>2} \trq = {:>9.2f} \tB[p] = {:>2d} "
        "\ttheta* = {:>5.4f} \tz = {:<9.2f}".format(entering_index + 1, entering_reduced_cost,
                                                    exiting_index + 1, step_length, updated_cost)
    )


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
