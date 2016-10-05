from numpy import random, pi, sin


def drop_needle():
    """
    To estimate a single drop, it is sufficient to randomize a location one end is dropped upon (between 0 and 1),
    an angle at which it is dropped, and to then check if it crosses either 0 or 1.

    :return: True if the simulated needle crossed at least one line, False otherwise
    """
    start = random.uniform()
    angle = random.uniform() * 2 * pi
    end = start + sin(angle)

    return end <= 0 or end >= 1


def pi_from_simulation_results(drops, hits):
    """
    Estimate pi from simulation results
    :param drops: the number of drops performed
    :param hits: the number of hits observed in simulation
    :return: the estimation for pi from these results
    """
    return 2.0 * drops / hits


def buffons_needle_approximation(n):
    """
    Estimate pi using Buffon's needle procedure
    :param n: the number of needles to drop
    :return: the approximation of pi from those drops
    """
    hits = 0
    for _ in xrange(n):
        hits += drop_needle()

    return pi_from_simulation_results(n, hits)


def consecutive_buffons_needle_approximation(n_values):
    """
    Running estimate of pi using Buffon's needle procedure, yielding values at every point in n_values
    :param n_values: the values of n to yield in
    :return: the approximation of pi from each n value drops
    """
    n = 0
    hits = 0

    for next_n in n_values:
        while n < next_n:
            hits += drop_needle()
            n += 1

        yield pi_from_simulation_results(n, hits)


def main():
    n_values = [10 ** x for x in xrange(2, 7)]
    for value in consecutive_buffons_needle_approximation(n_values):
        print value


if __name__ == '__main__':
    main()