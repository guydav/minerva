import numpy as np
from cs152 import memoize

# def max_swap_heuristic(state):
#     flattened_state = state.flatten()
#     n = flattened_state.shape[0]
#     locations = [0] * n  # 9 in the case of a 3x3
#     for i, x in enumerate(flattened_state):
#         locations[x] = i
#
#     steps = 0
#     while not np.all(flattened_state[:-1] <= flattened_state[1:]):
#         steps += 1
#         # The site's example uses n, where n is the number of tiles, 8 for a 3x3 puzzle
#         # I define n to be the size (n), and I'm 0-based, rather than 1, so subtract two.
#         first = locations[n - 1]
#         second = locations[locations[n - 1]]
#
#         # Swap elements
#         temp = flattened_state[first]
#         flattened_state[first] = flattened_state[second]
#         flattened_state[second] = temp
#
#         # Swap locations
#         temp = locations[flattened_state[first]]
#         locations[flattened_state[first]] = locations[flattened_state[second]]
#         locations[flattened_state[second]] = temp
#
#     return steps


@memoize.memoize_generic()
def _expected_row(k, n):
    if k >= n ** 2:
        raise ValueError('_expected_row expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))

    return np.floor(k / n)


@memoize.memoize_generic()
def _expected_column(k, n):
    if k >= n ** 2:
        raise ValueError('_expected_column expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))

    return k % n


@memoize.memoize_generic()
def _expected_location(k, n):
    """
    The expected location for k in an n x n grid, 0 <= k < n^2
    """
    if k >= n ** 2:
        raise ValueError('_expected_location expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))
    return _expected_row(k, n), _expected_column(k, n)


def _group_swap_test(groups, target_func):
    """
    A helper to the helper, testing that every element is in its correct group
    """
    n = groups.shape[0]
    return np.all([np.all([target_func(k, n) for k in groups[i]] == [i] * n)
                   for i in range(n)])


def _group_swap_suggest_move(groups, zero_group_index, move_group_index,
                             previous_move, target_func):
    """
    A helper to the helper, making a single group_swap move.
    The zero group is the one that includes the zero.
    The other one is the one that we consider swapping with.
    Direction is 1 when the zero group is before the other, and -1 otherwise.
    The previous move is which tile was swapped previously.
    """
    n = groups.shape[0]
    move_group = groups[move_group_index]
    move_direction = np.sign(move_group_index - zero_group_index)

    # We always wish to ignore the previous move we made, and it might be in the other group
    mask = [0] * n
    result = np.where(move_group == previous_move)[0]
    if result.size:
        mask[result[0]] = 1

    move_group_errors = np.ma.masked_array(list(map(lambda x, y: move_direction * (x - y),
                                                    [move_group_index] * n,
                                                    [target_func(k, n) for k in move_group])), mask)

    # apply the mask after calculating errors, to allow the target_func to cache
    # as np's masked_constant is unhashable
    masked_move_group_errors = np.ma.masked_array(move_group_errors, mask)
    # we return the index and benefit to allow the main helper to make the decision
    move_benefit = np.max(masked_move_group_errors.compressed())
    # if two values have the maximal, take the first one arbitrarily
    move_index = np.ma.where(masked_move_group_errors == move_benefit)[0][0]
    # returning the move group index to help me in the calling function
    return move_benefit, move_index, move_group_index


def _group_swap_do_swap(group, zero_index_tuple, move_index_tuple):
    # assert(abs(zero_index_tuple[0] - move_index_tuple[0]) != 1, 'Attempting illegal swap')
    group[zero_index_tuple], group[move_index_tuple] = group[move_index_tuple], group[zero_index_tuple]


@memoize.memoize_group_swap()
def _group_swap_helper(target_func, groups):
    """
    A helper to the x_y_heuristic, which handles solving the grouped problems.
    The groups are either rows or columns, and the target func receives the size
    of each puzzle side, and the value, and returns which group index (row or col)
    it should be in.
    """
    n = groups.shape[0]
    zero_group_index = int(target_func(np.argmin(groups), n))
    within_group_zero_index = np.argmin(groups[zero_group_index])
    previous_move = 0  # which tile was swapped in the previous move
    steps = 0

    while not _group_swap_test(groups, target_func):
        steps += 1
        # Only one direction
        if 0 == zero_group_index or n - 1 == zero_group_index:
            # If we're here, we're not solved, but have to move, so benefit doesn't matter
            move_group_index = 1 if 0 == zero_group_index else n - 2
            _, move_index, _ = \
                _group_swap_suggest_move(groups, zero_group_index, move_group_index,
                                         previous_move, target_func)

        else:
            # We have two moves we might make, let's make the best one
            suggested_moves = [_group_swap_suggest_move(groups, zero_group_index, suggestion_index,
                                                        previous_move, target_func)
                               for suggestion_index in (zero_group_index - 1, zero_group_index + 1)]

            _, move_index, move_group_index = max(suggested_moves)

        previous_move = groups[move_group_index][move_index]
        _group_swap_do_swap(groups, (zero_group_index, within_group_zero_index),
                            (move_group_index, move_index))
        zero_group_index, within_group_zero_index = move_group_index, move_index

    return steps






TEST_BOARDS = ([[2, 0, 6], [1, 3, 4], [7, 5, 8]],
               # [[5, 7, 6], [2, 4, 3], [8, 1, 0]],
               # [[7, 0, 8], [4, 6, 1], [5, 3, 2]],
               # [[2, 3, 7], [1, 8, 0], [6, 5, 4]],
               )


if __name__ == '__main__':
    print(_group_swap_helper(_expected_column, np.array(((0, 3, 6), (2, 5, 8), (1, 4, 7)))))

    # for board in TEST_BOARDS:
    #     print(_group_swap_helper(np.array(board), _expected_row))
    #     print(_group_swap_helper(np.array(board).T, _expected_column))


