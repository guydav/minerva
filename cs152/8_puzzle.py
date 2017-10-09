import numpy as np
from tabulate import tabulate
from cs152.heap import MinHeap  # OOP wrapper for Python's heapq I wrote at some point
import functools
from itertools import product, combinations
import operator
import timeit
from cs152 import memoize

timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

RANDOM_SEED = 33


@functools.total_ordering
class PuzzleNode:
    def __init__(self, state, parent=None, heuristic=lambda x: 0, blank_location=None):
        self.state = state
        self.parent = parent
        self.heuristic = heuristic

        if self.parent:
            self.path_cost = self.parent.path_cost + 1
        else:
            self.path_cost = 0

        if not blank_location:
            blank_location = np.unravel_index(np.argmin(self.state), self.state.shape)

        self.blank_location = blank_location

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return self.path_cost + self.heuristic(self.state) < other.path_cost + other.heuristic(other.state)

    def __str__(self):
        return tabulate(self.state, tablefmt='fancy_grid')

    def state_to_tuple(self):
        return tuple(int(x) for x in np.nditer(self.state))

    def generate_possible_moves(self):
        potential_steps = list(product((-1, 1), (0,))) + list(product((0,), (-1, 1)))
        next_nodes = []
        for step in potential_steps:
            try:
                next_blank_location = tuple(map(operator.add, self.blank_location, step))
                # Negative indexing is legal in numpy, but makes no sense in our case
                if any([x < 0 for x in next_blank_location]):
                    continue

                # the following line will throw an exception if the next location is invalid
                # we're perfectly okay with that
                current_next_blank_value = self.state[next_blank_location]

                next_state = self.state.copy()
                next_state[self.blank_location] = current_next_blank_value
                next_state[next_blank_location] = 0

                # avoid generating the parent state
                if self.parent and np.all(self.parent.state == next_state):
                    continue

                if np.all(next_state == np.array(((5, 6, 7), (2, 3, 4), (0, 1, 8)))):
                    print('*' * 20, 'SUGGESTED THE MOVE', '*' * 20)

                next_node = PuzzleNode(next_state, self, self.heuristic, next_blank_location)
                next_nodes.append(next_node)

            except IndexError:
                continue

        return next_nodes


SUCCESS_CODE = 0  # sometimes aptly named ERROR_OK, a name I've always found amusing
ERROR_CODE = -1
ERROR_TUPLE = (0, 0, ERROR_CODE)


def state_to_np_array(heuristic):
    def state_wrapper(state):
        if type(state) != np.ndarray:
            state = np.array(state)

        return heuristic(state)

    state_wrapper.__name__ = heuristic.__name__
    return state_wrapper


def _validate_initial_state(n, initial_state):
    if initial_state.shape != (n, n):
        print('Invalid initial state shape. Expected ({n}, {n}), found {shape}'.format(n=n, shape=initial_state.shape))

    expected = set(range(n ** 2))
    actual = set([int(x) for x in np.nditer(initial_state)])

    if expected == actual:
        return True

    expected_not_found = expected.difference(actual)
    actual_not_expected = actual.difference(expected)

    print('Initial state invalid:')
    if expected_not_found:
        print('Expected to find {s} and did not'.format(s=expected_not_found))
    if actual_not_expected:
        print('Found {s} and did not expect to'.format(s=actual_not_expected))

    return False


def _generate_goal_state(n):
    return np.array(range(n ** 2)).reshape((n, n))


def _print_solution(current, visited):
    solution = [current]

    while current.parent is not None:
        current = visited[current.parent.state_to_tuple()]
        solution.append(current)

    print('=' * 40 + ' PRINTING SOLUTION ' + '=' * 40)
    for step in reversed(solution):
        print(str(step))
        print(r'\/' * 7)


def single_tile_manhattan_distance(state, value):
    n = state.shape[0]
    index = tuple([x[0] for x in np.where(state == value)])
    return int(np.linalg.norm(list(map(operator.sub, index,
                                       _expected_location(value, n))), ord=1))


@state_to_np_array
def is_puzzle_solvable(initial_state):
    state = initial_state.copy().ravel()
    sign = 0
    index = 0
    while index < state.shape[0]:
        if state[index] == index:
            index += 1

        else:
            current = state[index]
            state[index], state[current] = state[current], state[index]
            sign += 1

    distance = single_tile_manhattan_distance(initial_state, 0)
    #     print(initial_state)
    #     print(state)
    #     print(sign, distance)
    return (sign % 2) == (distance % 2)


def solvePuzzle(n, initial_state, heuristic, print_solution=False, DEBUG=False):
    # validate n and initial_state
    if n < 2 or n > 128 or int(n) != n:
        print('Invalid n: {n}. Must be between 2 and 128 (inclusive)'.format(n=n))
        return ERROR_TUPLE

    if np.ndarray != type(initial_state):
        initial_state = np.array(initial_state)

    if not _validate_initial_state(n, initial_state):
        return ERROR_TUPLE

    if not is_puzzle_solvable(initial_state):
        print('Initial state is unsolvable by the test provided by Calabro (2005). Aborting...')
        return ERROR_TUPLE

    # set up search variables
    frontier = MinHeap()
    visited = {}
    num_steps = 0
    max_frontier = 0

    root = PuzzleNode(initial_state, heuristic=heuristic)
    frontier.push(root)
    visited[root.state_to_tuple()] = root
    goal = PuzzleNode(_generate_goal_state(n))

    # TODO: consider implementing IDA*

    while frontier:
        current = frontier.pop()

        if DEBUG:
            print('CURRENT MOVE:')

        if DEBUG:
            print(str(current))

        if np.all(current == goal):
            if print_solution:
                _print_solution(current, visited)

            return num_steps, max_frontier, 0

        # current will always be in visited - but perhaps we found a better way to get to it
        tuple_state = current.state_to_tuple()
        if tuple_state in visited and visited[tuple_state].path_cost < current.path_cost:
            current = visited[tuple_state]

        # generate next possible moves
        next_moves = current.generate_possible_moves()

        if DEBUG:
            print('NEXT MOVES')

        for move in next_moves:
            if np.all(move.state == np.array(((5, 6, 7), (2, 3, 4), (0, 1, 8)))):
                _print_solution(move, visited)
                return

            if DEBUG:
                print(move)

            tuple_state = move.state_to_tuple()
            # if not in visited, add to frontier, and add to visited
            if tuple_state not in visited:
                visited[tuple_state] = move
                frontier.push(move)

            # if in visited, check if should update visited min path cost
            else:
                if move.path_cost < visited[tuple_state].path_cost:
                    visited[tuple_state] = move

        num_steps += 1
        if max_frontier < len(frontier):
            max_frontier = len(frontier)

            #         if num_steps % 10000 == 0:
            #             print(num_steps, max_frontier, len(frontier))

    # If we arrive here, we've exhausted the frontier without finding a match
    return ERROR_TUPLE


@state_to_np_array
@memoize.memoize_heuristic()
def misplaced_tiles_heuristic(state):
    n = state.shape[0]
    return n ** 2 - np.sum(state == _generate_goal_state(n))


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


def _expected_location(k, n):
    """
    The expected location for k in an n x n grid, 0 <= k < n^2
    """
    if k >= n ** 2:
        raise ValueError('_expected_location expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))
    return _expected_row(k, n), _expected_column(k, n)


@state_to_np_array
@memoize.memoize_heuristic()
def manhattan_distance_heuristic(state):
    n = state.shape[0]
    return int(sum([np.linalg.norm(list(map(operator.sub, index,
                                            _expected_location(value, n))), ord=1)
                    for index, value in np.ndenumerate(state)]))


@state_to_np_array
@memoize.memoize_heuristic()
def linear_conflict_huristic(state):
    n = state.shape[0]
    conflicts = 0
    for row_num, row in enumerate(state):
        # Remove all elements that shouldn't be in the row
        correct_row = np.extract([np.floor(x / n) == row_num for x in row], row)
        if correct_row.shape[0] < 2:
            continue

        conflicts += sum(map(lambda x: x[0] > x[1], combinations(correct_row, 2)))

    return 2 * conflicts + manhattan_distance_heuristic(state)


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


# not memoized, since we pre-compute and memoize _group_swap_helper
@state_to_np_array
def x_y_heuristic(state):
    return _group_swap_helper(_expected_row, state) + \
           _group_swap_helper(_expected_column, state.T)


def recursive_generate_combinations(n, current=None, remainder=None, results=None):
    if current is None:
        current = []

    if remainder is None:
        remainder = set(range(n ** 2))

    if results is None:
        results = []

    if len(remainder) == n:
        current.append(tuple(remainder))
        results.append(tuple(current))

    else:
        for next_set in combinations(remainder, n):
            new_current = current[:]
            new_current.append(next_set)
            new_remainder = remainder.difference(next_set)
            recursive_generate_combinations(n, new_current, new_remainder, results)

    return results


def precompute_x_y_heuristic(n):
    for groups in filter(is_puzzle_solvable, recursive_generate_combinations(n)):
        for target_func in (_expected_row, _expected_column):
            _group_swap_helper(target_func, groups)


heuristics = (misplaced_tiles_heuristic, manhattan_distance_heuristic,
              linear_conflict_huristic, x_y_heuristic)


TEST_BOARDS = ([[5, 7, 6], [2, 4, 3], [8, 1, 0]], #  )
               [[7, 0, 8], [4, 6, 1], [5, 3, 2]],
               [[2, 3, 7], [1, 8, 0], [6, 5, 4]])


if __name__ == '__main__':
    print('Precomputing for n = 3')

    def timed_precompute():
        precompute_x_y_heuristic(3)

    timer = timeit.Timer(timed_precompute)
    length = timer.timeit(number=1)[0]
    print('Precomputing took', length, 'seconds')

    # board = [[5, 7, 6], [2, 4, 3], [8, 1, 0]]
    # print(solvePuzzle(len(board), board, x_y_heuristic))

    all_results = []
    for board, current_heuristic in product(TEST_BOARDS, heuristics):
        print('Running on:', board, 'with heuristic:', current_heuristic.__name__)

        def foo():
            solvePuzzle(len(board), board, current_heuristic, print_solution=False, DEBUG=False)

        timer = timeit.Timer(foo, 'gc.enable()')
        length, current_result = timer.timeit(number=1)
        print(current_result, length)
        all_results.append(current_result)
