import math
from scipy import stats


def normal_cdf(x, n=100):
    """
    :param x: The value for which we want to calculate the CDF
    :param n: The number of rounds to use in the calculation
    :return: The standard normal CDF for this value of X
    """
    total = x
    value = x

    for i in xrange(1, n):
        value *= (x * x) / (2 * i + 1)
        total += value

    return 0.5 + math.exp((-1 * x ** 2) / 2) * total / ((2 * math.pi) ** 0.5)


def test_cdf(x):
    return stats.norm.cdf(x) - normal_cdf(x)


if __name__ == '__main__':
    x = -3.0
    while x <= 3.0:
        print x, test_cdf(x)
        x += 0.1
