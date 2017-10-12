import numpy as np

from tabulate import tabulate
from cs152.heap import MinHeap  # OOP wrapper for Python's heapq I wrote at some point
import functools
from itertools import product, combinations
import operator
import timeit
from cs152 import memoize

# Change timeit's template to return the value from the function
# From https://stackoverflow.com/questions/24812253/how-can-i-capture-return-value-with-python-timeit-module
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

######################################################################################


@functools.total_ordering
class PuzzleNode:
    def __init__(self, state, parent=None, heuristic=lambda x: 0, blank_location=None):
        """
        Initialize a new puzzle node
        :param state: The state of the puzzle at this point in time, an n x n numpy array
        :param parent: The parent puzzle node for this one, None if this is the initial
        :param heuristic: Which heuristic to use, by default the null heuristic, which
            makes the A* simply BFS
        :param blank_location: the current location of the blank tile, to avoid searching
            for it every time
        """
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
        return np.array_equal(self.state, other.state)

    def __lt__(self, other):
        """
        Use the less than function and functools's total_ordering decorator to allow the minheap
         to compare puzzle nodes to each other
        :param other: The other puzzle node to compare to
        :return:
        """
        return self.path_cost + self.heuristic(self.state) < other.path_cost + other.heuristic(other.state)

    def __str__(self):
        return tabulate(self.state, tablefmt='fancy_grid')

    def state_to_tuple(self):
        return tuple(int(x) for x in np.nditer(self.state))

    def generate_possible_moves(self):
        """
        Generate the next moves possible from this one. Excludes any where the index
        :return:
        """
        n = self.state.shape[0]
        potential_steps = list(product((-1, 1), (0,))) + list(product((0,), (-1, 1)))
        next_nodes = []
        for step in potential_steps:
            next_blank_location = tuple(map(operator.add, self.blank_location, step))
            # Negative indexing is legal in numpy, but makes no sense in our case
            if any([x < 0 or x >= n for x in next_blank_location]):
                continue

            current_next_blank_value = self.state[next_blank_location]

            next_state = self.state.copy()
            next_state[self.blank_location] = current_next_blank_value
            next_state[next_blank_location] = 0

            # avoid generating the parent state
            if self.parent and np.array_equal(self.parent.state, next_state):
                continue

            next_node = PuzzleNode(next_state, self, self.heuristic, next_blank_location)
            next_nodes.append(next_node)

        return next_nodes


######################################################################################


SUCCESS_CODE = 0  # sometimes aptly named ERROR_OK, a name I've always found amusing
ERROR_CODE = -1
ERROR_TUPLE = (0, 0, ERROR_CODE)


def state_to_np_array(heuristic):
    """
    A neat little wrapper for a heuristic function (or any other function receiving only the state)
    that verifies that the state it receives is an np.array
    :param heuristic: The function to wrap
    :return: The function, wrapped in verification that it receives an np.array
    """
    def state_wrapper(state):
        if type(state) != np.ndarray:
            state = np.array(state)

        return heuristic(state)

    state_wrapper.__name__ = heuristic.__name__
    return state_wrapper


def _validate_initial_state(n, initial_state):
    """
    Validate the initial state, verifying that it is indeed a valid n-puzzle.
    :param n: The size of each side of the puzzle
    :param initial_state: The initial state to verify
    :return: True if it's a valid initial state, False otherwise
    """
    if initial_state.shape != (n, n):
        print('Invalid initial state shape. Expected ({n}, {n}), found {shape}'.format(n=n,
                                                                                       shape=initial_state.shape))

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
        print(r'\/' * 6)


def single_tile_manhattan_distance(state, value):
    """
    The manhattan distance heuristic, but implemented for a single value,
    rather than over the entire state, to help in the solvability test
    :param state: The state to compute distance in
    :param value: The value of the title to compute distance for
    :return: The manhattan distance between the tile and its final location
    """
    n = state.shape[0]
    index = tuple([x[0] for x in np.where(state == value)])
    return int(np.linalg.norm(list(map(operator.sub, index,
                                       _expected_location(value, n))), ord=1))


@state_to_np_array
def is_puzzle_solvable(initial_state):
    """
    Implementation of Calabro's (2005) algorithm for determining state solvability.
    As discussed above, we examine the number of swaps required to get all tiles in
    their correct places, and compare the sign (mod 2) to the manhattan distance
    of the blank tile.
    :param initial_state: The initial state to examine
    :return: True if it's solveable, False otherwise
    """
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
    return (sign % 2) == (distance % 2)


INITIAL_MAX_DEPTH = 10
MAX_FRONTIER_SIZE_TO_RETAIN = 10 ** 5


def solvePuzzle(n, initial_state, heuristic, print_solution=False,
                iterative_deepening=False, initial_max_depth=INITIAL_MAX_DEPTH,
                max_frontier_size_to_retain=MAX_FRONTIER_SIZE_TO_RETAIN, debug=False):
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
    root = PuzzleNode(initial_state, heuristic=heuristic)
    frontier = MinHeap(root)

    if iterative_deepening:
        max_depth_frontier = MinHeap()
        max_depth = initial_max_depth

    visited = {}
    num_steps = 0
    max_frontier = 0

    visited[root.state_to_tuple()] = root
    goal = PuzzleNode(_generate_goal_state(n))

    while (not iterative_deepening and len(frontier) > 0) or \
          (iterative_deepening and (len(frontier) > 0 or len(max_depth_frontier) > 0)):
        # Exhausted current depth, move onto next one
        if 0 == len(frontier):
            if debug:
                print('Exhausted the previous max depth, moved onto d = {md}'.format(md=max_depth))

            # Test if we can keep where we were going from, or must restart anew
            if len(max_depth_frontier) < max_frontier_size_to_retain:
                frontier = max_depth_frontier

            else:
                frontier = MinHeap(root)
                visited = {}

            max_depth_frontier = MinHeap()
            max_depth += 1

        current = frontier.pop()

        if debug and num_steps % 10000 == 0:
            print(current.heuristic(current.state), current.path_cost)
            print(str(current))

        if current == goal:
            if print_solution:
                _print_solution(current, visited)

            return num_steps, max_frontier, SUCCESS_CODE

        # current will always be in visited - but perhaps we found a better way to get to it
        tuple_state = current.state_to_tuple()
        if tuple_state in visited and visited[tuple_state].path_cost < current.path_cost:
            current = visited[tuple_state]

        # check iterative deepening condition
        if iterative_deepening and current.path_cost >= max_depth:
            max_depth_frontier.push(current)
            continue

        # generate next possible moves
        next_moves = current.generate_possible_moves()

        for move in next_moves:
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

    # If we arrive here, we've exhausted the frontier without finding a match
    return ERROR_TUPLE


######################################################################################


@state_to_np_array
@memoize.memoize_heuristic
def misplaced_tiles_heuristic(state):
    n = state.shape[0]
    return n ** 2 - np.sum(np.equal(state, _generate_goal_state(n)))


@memoize.minimal_memoize
def _expected_row(k, n):
    if k >= n ** 2:
        raise ValueError('_expected_row expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))

    return np.floor(k / n)


@memoize.minimal_memoize
def _expected_column(k, n):
    if k >= n ** 2:
        raise ValueError('_expected_column expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))

    return k % n


@memoize.minimal_memoize
def _expected_location(k, n):
    """
    The expected location for k in an n x n grid, 0 <= k < n^2.
    Broken down into two utility functions, one handling the row, the other the column.
    """
    if k >= n ** 2:
        raise ValueError(
            '_expected_location expects 0 <= k < n^2, received k={k} and n={n}'.format(k=k, n=n))
    return _expected_row(k, n), _expected_column(k, n)


@state_to_np_array
@memoize.memoize_heuristic
def manhattan_distance_heuristic(state):
    """
    The manhattan distance over every tile and its current location, using the
    _expected_location helper method and some Python trickery.
    :param state: The state to evaluate the distance for
    :return: The total manhattan distance between the state and its goal state
    """
    n = state.shape[0]
    return int(sum([np.linalg.norm(list(map(operator.sub, index,
                                            _expected_location(value, n))), ord=1)
                    for index, value in np.ndenumerate(state)]))


def _linear_conflict_helper(state, target_func):
    """
    This helper actually evaluates the number of conflicts within each line.
    A line, in this sense, is either a row or a column, based on whether the
    state is passed in, or its transpose. The target function should be either
    _expected_row or _expected_column, and it helps identify which elements
    are in their proper line.
    :param state: The current state to evaluate (or its transpose for columns)
    :param target_func: Either _expected_row or _expected_column
    :return: The number of conflicts encountered
    """
    n = state.shape[0]
    conflicts = 0
    for line_num, line in enumerate(state):
        # Remove all elements that shouldn't be in this line
        in_correct_line = np.extract([target_func(x, n) == line_num for x in line], line)
        if in_correct_line.shape[0] < 2:
            continue

        conflicts += sum(map(lambda x: x[0] > x[1], combinations(in_correct_line, 2)))

    return conflicts


@state_to_np_array
@memoize.memoize_heuristic
def linear_conflict_heuristic(state):
    """
    The linear conflict heuristic, a manhattan distance also taking into account any
     colinear conflict, both within a row and within a column, since both must be resolved.
    :param state: The state to evaluate
    :return: The linear conflict heuristic value - the manhattan distance plus
        twice the number of conflicts found
    """
    conflicts = _linear_conflict_helper(state, _expected_row) + \
                _linear_conflict_helper(state.T, _expected_column)
    return 2 * conflicts + manhattan_distance_heuristic(state)


######################################################################################


def _group_swap_test(groups, target_func):
    """
    A helper to the helper, testing that every element is in its correct group
    """
    n = groups.shape[0]
    return np.all([np.array_equal([target_func(k, n) for k in groups[i]], [i] * n)
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
    group[zero_index_tuple], group[move_index_tuple] = group[move_index_tuple], group[zero_index_tuple]


@memoize.memoize_group_swap
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
    """
    Recursively generate all combinations of rows or columns matching a particular
    board size, such as to generate all possible boards.

    This could probably be made more elegant by turning it into a generator

    :param n: The board size (side length) to generate
    :param current: The current collection rows being expanded
    :param remainder: The elements we're still trying to allocate to rows
    :param results: The combinations of rows encountered so far.
    :return: all board combinations for the given size
    """
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
    """
    Pre-compute and fill the cache for the x-y heuristic's helper,
    _group_swap_helper, for a given board size
    :param n: The size of the board (side length) to precompute
    :return: None; cache for _group_swap_helper filled
    """
    print('Precomputing for n = {n}'.format(n=n))

    def timed_precompute():
        # No need to precompute unsolvable puzzles
        for groups in filter(is_puzzle_solvable, recursive_generate_combinations(n)):
            for target_func in (_expected_row, _expected_column):
                _group_swap_helper(target_func, groups)

    precompute_timer = timeit.Timer(timed_precompute)
    precompute_length = precompute_timer.timeit(number=1)[0]
    print('Precomputing took', precompute_length, 'seconds')


######################################################################################


TEST_BOARDS = ([[5, 7, 6], [2, 4, 3], [8, 1, 0]],
               [[7, 0, 8], [4, 6, 1], [5, 3, 2]],
               [[2, 3, 7], [1, 8, 0], [6, 5, 4]],
              )

heuristics = (misplaced_tiles_heuristic, manhattan_distance_heuristic,
              linear_conflict_heuristic, x_y_heuristic)


def run_test_boards(test_boards=TEST_BOARDS, heuristic_list=heuristics,
                    print_solution=False, iterative_deepening=False, debug=False):
    results = []

    for board, heuristic in product(test_boards, heuristic_list):
        print('Running on:', board, 'with heuristic:', heuristic.__name__)

        def solve():
            return solvePuzzle(len(board), board, heuristic, print_solution=print_solution,
                               iterative_deepening=iterative_deepening, debug=debug)

        timer = timeit.Timer(solve, 'gc.enable()')
        length, result = timer.timeit(number=1)
        print(result, '{length:.3f} seconds'.format(length=length))
        results.append((result, length))

    return results


def pretty_print_results(results, test_boards=TEST_BOARDS, heuristic_list=heuristics):
    textual_results = [', '.join([str(x) for x in r] + ['{l:.3f}s'.format(l=l)]) for r, l in results]

    reshaped_results = np.array(textual_results).reshape((len(test_boards), len(heuristic_list)))
    headers = [heuristic.__name__.replace('memoized', '').replace('(', '').replace(')', '')
               for heuristic in heuristic_list]

    print(tabulate(reshaped_results, headers, tablefmt='fancy_grid'))


ROUJIAS_4_BY_4_BOARDS = (
    [[1, 2, 3, 4], [10, 8, 12, 6], [14, 5, 0, 15], [9, 13, 7, 11]],
    [[7, 2, 4, 9], [5, 10, 0, 6], [8, 11, 3, 1], [12, 13, 14, 15]],
    # [[13, 9, 7, 15], [3, 6, 8, 4], [11, 10, 2, 12], [5, 14, 1, 0]],
    # [[15, 2, 1, 12], [8, 5, 6, 11], [4, 9, 10, 7], [3, 14, 13, 0]],
)


######################################################################################


if __name__ == '__main__':
    run_test_boards(TEST_BOARDS[1:],
                    heuristic_list=[linear_conflict_heuristic],
                    print_solution=True)

    # print('Regular A*:')
    # regular_results = run_test_boards()
    # pretty_print_results(regular_results)
    #
    # print('IDA*')
    # id_results = run_test_boards(iterative_deepening=True, debug=False)
    # pretty_print_results(id_results)

    # four_by_four_results = run_test_boards(test_boards=ROUJIAS_4_BY_4_BOARDS,
    #                                        heuristic_list=(linear_conflict_heuristic,))
    # pretty_print_results(four_by_four_results, test_boards=ROUJIAS_4_BY_4_BOARDS,
    #                      heuristic_list=(linear_conflict_heuristic,))
    #
