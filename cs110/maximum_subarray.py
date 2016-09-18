from profilehooks import profile
from numpy import random


@profile(sort='calls', immediate=False)
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


@profile(sort='calls', immediate=False)
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


def incremental_max_subarray(new_list, previous_max_subarray):
    """
    Compute the maximum subarray of a list to which a new number was appended to, given
    that we know the subarray of the list without the new element.
    :param new_list: The list to work on
    :param previous_max_subarray: The maximum subarray of new_list[:-1]
    :return: the maximum subarray of the new list
    """
    previous_max, previous_left, previous_right = previous_max_subarray
    list_length = len(new_list)
    new_value = new_list[-1]

    # First case - the maximum subarray included the last day
    # all we need to do is see if adding the new last day helps it
    # This entire branch is O(1)
    if previous_right == list_length - 1:
        last_two_days_difference = new_value - new_list[-2]
        if last_two_days_difference > 0:
            return previous_max + last_two_days_difference, previous_left, previous_right + 1

        return previous_max_subarray

    # Otherwise, the only way the new maximum subarray could be better is if it included the last day
    # We also know that the minimum of the list must the last day or before it.
    # If it is the last day, it cannot be part of a new maximum subarray, so we're done.
    min_value = min(new_list)  # min is O(n)
    if min_value == new_value:
        return previous_max_subarray

    # Otherwise, we now check if the last day - the minimum value is the new maximum subarray
    max_with_new_value = new_value - min_value
    if max_with_new_value > previous_max:
        return max_with_new_value, new_list.index(min_value), list_length - 1  # list.index() is O(n)

    return previous_max_subarray


def main():
    # 2.2

    # One example
    ints = [i for i in random.random_integers(-10, 10, 10)]
    diffs = [ints[i] - ints[i - 1] for i in xrange(1, len(ints))]
    # print ints
    # print diffs
    # print brute_force(ints)
    print ints
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

