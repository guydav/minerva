import numpy
import heapq
import tabulate


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


def extended_bottom_up_cut_rod(n, prices=DEFAULT_PRICES, c=0):
    if n <= 0:
        return [0], [0]

    r_cache = [0] * (n + 1)
    s_cache = [0] * (n + 1)

    for j in xrange(1, n + 1):
        max_price = NEGATIVE_INFINITY

        for i in xrange(1, min(j + 1, len(prices))):

            new_price = prices[i] + r_cache[j - i] - c
            if max_price < prices[i] + r_cache[j - i]:
                max_price = new_price
                s_cache[j] = i

        r_cache[j] = max_price

    return r_cache, s_cache


def print_extended_bottom_up_cut_rod(n, prices=DEFAULT_PRICES, c=0):
    r_cache, s_cache = extended_bottom_up_cut_rod(n, prices, c)
    while n > 0:
        current_cut = s_cache[n]
        print current_cut, " ",
        n -= current_cut

    print


def bottom_up_matrix_multiplication_one_based(p):
    n = len(p) - 1
    m = numpy.zeros((n, n))
    s = numpy.zeros((n - 1, n))

    for l in xrange(2, n + 1):
        for i in xrange(1, n - l + 2):
            j = i + l - 1
            m[i - 1][j - 1] = float("inf")

            for k in xrange(i, j):
                q = m[i - 1][k - 1] + m[k][j - 1] + p[i - 1] * p[k] * p[j]
                if q < m[i - 1][j - 1]:
                    m[i - 1][j - 1] = q
                    s[i - 1][j - 1] = int(k)

    return m, s


def print_optimal_matrix_one_based(s, i, j):
    if i == j:
        print 'A{i}'.format(i=i),

    else:
        print '(',
        print_optimal_matrix_one_based(s, i, int(s[i - 1][j - 1]))
        print_optimal_matrix_one_based(s, int(s[i - 1][j - 1]) + 1, j)
        print ')',


def bottom_up_matrix_multiplication(p):
    n = len(p) - 1
    m = numpy.zeros((n, n))
    s = numpy.zeros((n - 1, n))

    for l in xrange(2, n + 1):
        for i in xrange(n - l + 1):
            j = i + l - 1
            m[i][j] = float("inf")

            for k in xrange(i, j):
                q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = int(k)

    return m, s


def print_optimal_matrix(s, i, j):
    if i == j:
        print 'A{i}'.format(i=i + 1),

    else:
        k = int(s[i][j])
        print '(',
        print_optimal_matrix(s, i, k)
        print_optimal_matrix(s, k + 1, j)
        print ')',


def memoized_matrix_chain_one_based(p):
    n = len(p) - 1
    m = numpy.ones((n, n))
    m *= float('inf')
    return lookup_chain_one_based(m, p, 1, n)


def lookup_chain_one_based(m, p, i, j):
    if m[i - 1][j - 1] < float('inf'):
        return m[i - 1][j - 1]

    if i == j:
        m[i - 1][j - 1] = 0

    else:
        for k in xrange(i, j):
            q = lookup_chain_one_based(m, p, i, k) + \
                lookup_chain_one_based(m, p, k + 1, j) + \
                p[i - 1] * p[k] * p[j]

            if q < m[i - 1][j - 1]:
                m[i - 1][j - 1] = q

    return m[i - 1][j - 1]


def memoized_matrix_chain(p):
    n = len(p) - 1
    m = numpy.full((n, n), float('inf'))
    return lookup_chain(m, p, 0, n - 1)


def lookup_chain(m, p, i, j):
    if m[i][j] < float('inf'):
        return m[i][j]

    if i == j:
        m[i][j] = 0

    else:
        for k in xrange(i, j):
            q = lookup_chain(m, p, i, k) + \
                lookup_chain(m, p, k + 1, j) + \
                p[i] * p[k + 1] * p[j + 1]

            if q < m[i][j]:
                m[i][j] = q

    return m[i][j]


def matrix_multiplication_testing():
    p = [30, 35, 15, 5, 10, 20, 25]
    ob_m, ob_s = bottom_up_matrix_multiplication_one_based(p)
    m, s = bottom_up_matrix_multiplication(p)

    # print_optimal_matrix(s, 0, 5)
    print memoized_matrix_chain(p)


def currency_trading_djikstra(rates, start_node, finish_node):
    frontier = []
    heapq.heappush(frontier, (-1 * rates[start_node][start_node], start_node, [start_node]))
    results = {}

    optimal_price = 0
    optimal_path = None

    while frontier:
        current_price, current_node, current_path = heapq.heappop(frontier)
        current_price *= -1

        for next_node in xrange(len(rates)):
            if (current_node == next_node) or (next_node == start_node):
                continue

            next_price = current_price * rates[current_node][next_node]
            next_key = (current_path[0], next_node)

            if (next_key not in results) or results[next_key][1] + 0.01 < next_price:
                next_path = current_path + [next_node]
                results[next_key] = (next_path, next_price)

                if next_node == finish_node:
                    if next_price > optimal_price + 0.01:
                        optimal_price = next_price
                        optimal_path = next_path
                        print optimal_price, optimal_path

                else:
                    heapq.heappush(frontier, (-1 * next_price, next_node, next_path,))

    return optimal_price, optimal_path


def currency_trading_djikstra_log(rates, start_node, finish_node):


    frontier = []
    heapq.heappush(frontier, (-1 * rates[start_node][start_node], start_node, [start_node]))
    results = {}

    optimal_price = 0
    optimal_path = None

    while frontier:
        current_price, current_node, current_path = heapq.heappop(frontier)
        current_price *= -1

        for next_node in xrange(len(rates)):
            if (current_node == next_node) or (next_node == start_node):
                continue

            next_price = current_price * rates[current_node][next_node]
            next_key = (current_path[0], next_node)

            if (next_key not in results) or results[next_key][1] + 0.01 < next_price:
                next_path = current_path + [next_node]
                results[next_key] = (next_path, next_price)

                if next_node == finish_node:
                    if next_price > optimal_price + 0.01:
                        optimal_price = next_price
                        optimal_path = next_path
                        print optimal_price, optimal_path

                else:
                    heapq.heappush(frontier, (-1 * next_price, next_node, next_path,))

    return optimal_price, optimal_path


def test_currency_trading():
    rates = numpy.array([numpy.fromstring('1 0.741 0.657 1.061 1.005', sep=' '),
                         numpy.fromstring('1.349 1 0.888 1.433 1.366', sep=' '),
                         numpy.fromstring('1.521 1.126 1 1.614 1.538', sep=' '),
                         numpy.fromstring('0.942 0.698 0.619 1 0.953', sep=' '),
                         numpy.fromstring('0.995 0.732 0.650 1.049 1', sep=' ')])

    print currency_trading_djikstra(rates, 1, 4)


def memoized_game_strategy(coins, start=0, end=None, cache=None, start_player=1):
    if cache is None:
        cache = {}

    if end is None:
        end = len(coins) - 1

    key = (start, end)
    if key in cache:
        return cache[key]

    if start == end:
        value = coins[start] * start_player

    elif start + 1 == end:
        value = numpy.abs(coins[start] - coins[end]) * start_player

    else:
        options = (start_player * coins[start] + memoized_game_strategy(coins, start + 1, end, cache, -1 * start_player),
                   start_player * coins[end] + memoized_game_strategy(coins, start, end - 1, cache, -1 * start_player))

        if 1 == start_player:
            value = max(options)

        else:
            value = min(options)

    cache[key] = value
    return value


def bottom_up_game_strategy(coins):
    cache = {}
    n = len(coins)

    for game_length in xrange(1, n + 1):
        start_player = (game_length % 2 == n % 2) and 1 or -1

        for start in xrange(n - game_length + 1):
            end = start + game_length - 1

            key = (start, end)
            if 1 == game_length:
                value = coins[start] * start_player

            elif 2 == game_length:
                value = abs(coins[start] - coins[end]) * start_player

            else:
                options = (start_player * coins[start] + cache[(start + 1, end)],
                           start_player * coins[end] + cache[(start, end - 1)])

                if 1 == start_player:
                    value = max(options)

                else:
                    value = min(options)

            cache[key] = value

    print cache

    return cache[(0, n - 1)]


def test_game_strategy():
    coins = [2, 10, 1, 5, 3]
    print memoized_game_strategy(coins)
    print bottom_up_game_strategy(coins)


def bottom_up_longest_subsequence(x, y):
    m = len(x)
    n = len(y)

    cache = numpy.zeros((m + 1, n + 1))
    subsequences = numpy.chararray((m + 1, n + 1))
    subsequences[:] = ''

    for i in xrange(1, m + 1):
        for j in xrange(1, n + 1):
            up = cache[i - 1][j]
            left = cache[i][j - 1]

            if x[i - 1] == y[j - 1]:
                cache[i][j] = cache[i - 1][j - 1] + 1
                subsequences[i][j] = '\\'

            elif up >= left:
                cache[i][j] = up
                subsequences[i][j] = '^'
            else:
                cache[i][j] = left
                subsequences[i][j] = '<'

    return cache, subsequences


def print_longest_subsequence(b, x, i, j):
    if 0 == i or 0 == j:
        return

    current = b[i][j]
    if '\\' == current:
        print_longest_subsequence(b, x, i - 1, j - 1)
        print x[i - 1],

    elif '^' == current:
        print_longest_subsequence(b, x, i - 1, j)

    else:
        print_longest_subsequence(b, x, i, j - 1)


def print_longest_subsequence_from_table(c, x, y, i, j):
    if 0 == i or 0 == j:
        return

    if x[i - 1] == y[j - 1]:
        print_longest_subsequence_from_table(c, x, y, i - 1, j - 1)
        print x[i - 1],

    elif c[i - 1, j] >= c[i, j - 1]:
        print_longest_subsequence_from_table(c, x, y, i - 1, j)

    else:
        print_longest_subsequence_from_table(c, x, y, i, j - 1)


def test_longest_subsequence():
    x = '10010101'  # 'abcbdab'
    y = '010110110' # 'bdcaba'
    m = len(x)
    n = len(y)

    c, b = bottom_up_longest_subsequence(x, y)
    print_longest_subsequence(b, x, m, n)
    print
    print_longest_subsequence_from_table(c, x, y, m, n)


def longest_monotonous_subsequence(x):
    n = len(x)
    length_cache = [1] * n
    transition_table = [-1] * n

    max_length = 1
    end_index = 0

    for i in xrange(1, n):
        current_max_j = -1
        current_max_length = 1

        for j in xrange(i - 1, -1, -1):
            if x[i] > x[j] and length_cache[j] >= current_max_length:
                current_max_j = j
                current_max_length = length_cache[j]

        if -1 != current_max_j:
            new_length = current_max_length + 1
            length_cache[i] = new_length
            transition_table[i] = current_max_j

            if new_length > max_length:
                end_index = i
                max_length = new_length

    return length_cache, transition_table, end_index


def print_longest_monotonous_subsequence(x, transition_table, end_index):
    if 0 > end_index:
        return

    print_longest_monotonous_subsequence(x, transition_table, transition_table[end_index])
    print x[end_index],


def test_monotonous_subsequence():
    x = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    length_cache, transition_table, end_index = longest_monotonous_subsequence(x)
    print tabulate.tabulate([x, length_cache, transition_table], headers=range(len(x)))
    print_longest_monotonous_subsequence(x, transition_table, end_index)


if __name__ == '__main__':
    # matrix_multiplication_testing()
    # test_currency_trading()
    # test_game_strategy()
    # test_longest_subsequence()
    test_monotonous_subsequence()