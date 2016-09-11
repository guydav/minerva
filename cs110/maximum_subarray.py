from profilehooks import profile
from numpy import random


@profile(sort='calls', immediate=True)
def brute_force(price_list):
    buy_index = 0
    sell_index = 0
    max_profit = 0

    length = len(price_list)

    for i in xrange(length - 1):
        for j in xrange(i + 1, length):
            profit = price_list[j] - price_list[i]
            if profit > max_profit:
                max_profit = profit
                buy_index = i
                sell_index = j

    return max_profit, buy_index, sell_index


@profile(sort='calls', immediate=True)
def divide_and_conquer(diff_list, low=0, high=None):
    if high is None:
        high = len(diff_list)

    if low + 1 == high:
        return diff_list[low], low, high

    mid = low + (high - low) / 2
    results = [divide_and_conquer(diff_list, low, mid),
               divide_and_conquer(diff_list, mid, high),
               find_crossing_subarray(diff_list, low, mid, high)]

    return max(results)


def find_crossing_subarray(diff_list, low, mid, high):
    current_sum = 0
    left_sum = None
    left_index = mid

    for l in xrange(mid, low - 1, -1):
        current_sum += diff_list[l]

        if left_sum is None or current_sum > left_sum:
            left_sum = current_sum
            left_index = l

    current_sum = 0
    right_sum = 0
    right_index = mid + 1

    for r in xrange(mid + 1, high):
        current_sum += diff_list[r]
        if right_sum is None or current_sum > right_sum:
            right_sum = current_sum
            right_index = r

    # print low, mid, high, '|', left_sum + right_sum, left_index, right_index

    return left_sum + right_sum, left_index, right_index  # to account for it being an exclusive interval


def main():
    # 2.2

    # One example
    ints = [i for i in random.random_integers(-100, 100, 1000)]
    diffs = [ints[i] - ints[i - 1] for i in xrange(1, len(ints))]
    # print ints
    # print diffs
    print brute_force(ints)
    print divide_and_conquer(diffs)

    # Random testing
    # for i in xrange(100):
    #     ints = [i for i in random.random_integers(-100, 100, 100)]
    #     diffs = [ints[i] - ints[i - 1] for i in xrange(1, len(ints))]
    #
    #     bf = brute_force(ints)
    #     dac = divide_and_conquer(diffs)
    #
    #     if bf[0] != dac[0]:
    #         print ints
    #         print 'Brute force:', bf
    #         print 'Divide and conquer:', dac

if __name__ == '__main__':
    main()

