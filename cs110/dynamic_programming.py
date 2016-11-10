from functools import partial
from timeit import timeit
from numpy.random import randint

NEGATIVE_INFINITY = float("-inf")
DEFAULT_PRICES = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]


def recursive_fibonacci(n):
    if 1 >= n:
        return 1

    return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)


def memoized_fibonacci(n, cache=None):
    if cache is None:
        cache = [0] * n
        cache[0] = 1
        cache[1] = 1

    if 1 >= n:
        return 1

    if 0 != cache[n]:
        return cache[n]

    result = memoized_fibonacci(n - 1, cache) + memoized_fibonacci(n - 2, cache)
    cache[n] = result
    return result


def bottom_up_fibonnaci(n):
    if n == 0 or n == 1:
        return 1

    # Could avoid saving intermediate values, but I already have this saved
    cache = [1, 1]
    for i in xrange(2, n):
        cache.append(cache[i - 1] + cache[i - 2])

    return cache[n - 1] + cache[n - 2]


def cut_rod(n, prices=DEFAULT_PRICES):
    if n <= 0:
        return 0

    max_price = NEGATIVE_INFINITY
    for i in xrange(n):
        max_price = max(max_price, prices[i + 1] + cut_rod(n - i - 1, prices))

    return max_price


def memoized_cut_rod(n, prices=DEFAULT_PRICES, cache=None):
    if cache is None:
        cache = [NEGATIVE_INFINITY] * (n + 1)
        cache[0] = 0

    if n <= 0:
        return 0

    if NEGATIVE_INFINITY != cache[n]:
        return cache[n]

    max_price = NEGATIVE_INFINITY
    for i in xrange(n):
        max_price = max(max_price, prices[i + 1] + memoized_cut_rod(n - i - 1, prices, cache))

    cache[n] = max_price
    return max_price


def bottom_up_cut_rod(n, prices=DEFAULT_PRICES):
    if n <= 0:
        return 0

    cache = [0] * (n + 1)

    for j in xrange(1, n + 1):
        max_price = NEGATIVE_INFINITY

        for i in xrange(1, j + 1):
            max_price = max(max_price, prices[i] + cache[j - i])

        cache[j] = max_price

    return cache[n]


def extended_bottom_up_cut_rod(n, prices=DEFAULT_PRICES):
    if n <= 0:
        return [0], [0]

    r_cache = [0] * (n + 1)
    s_cache = [0] * (n + 1)

    for j in xrange(1, n + 1):
        max_price = NEGATIVE_INFINITY

        for i in xrange(1, j + 1):
            new_price = prices[i] + r_cache[j - i]
            if max_price < prices[i] + r_cache[j - i]:
                max_price = new_price
                s_cache[j] = i

        r_cache[j] = max_price

    return r_cache, s_cache


def print_extended_bottom_up_cut_rod(n, prices=DEFAULT_PRICES):
    r_cache, s_cache = extended_bottom_up_cut_rod(n, prices)
    while n > 0:
        current_cut = s_cache[n]
        print current_cut, " ",
        n -= current_cut

    print
