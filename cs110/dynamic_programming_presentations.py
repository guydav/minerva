# Instructions:
# -------------
#  1. This is a valid python file. Update the code at the bottom to include
#     your student email address.
#  2. Run the code to find out which one of the problems you need to do.
#  3. Solve your assigned problem.  Since the assignment is deterministic, you
#     can also find out who else is working on the same problem.  Feel free to
#     approach other students if you get stuck conceptually, but write your
#     code by yourself.
#  4. Now answer the following questions on your solution
#     a) Define variables which will be useful in determining the computational
#        complexity of your solution.
#     b) How does it scale in terms of time (using your variables from a)?
#     c) How does it scale in terms of space (using your variables from a)?
#     d) Each problem has an example so that you can see how to structure your
#        solution.  However the answers to the examples are wrong.  What is the
#        actual answer for your example documentation?
#
# Directions:
# -----------
#
# You and your friends rented a bus and have gotten lost.  This is because the
# person with the map was sitting at the back of the bus and gave directions to
# the person in front of them.  This person then told the person in front of them.
# Eventually the directions reached the driver in the front.  Occasionally someone
# would make a mistake, and they would either:
#  - leave out a step,
#  - add a step,
#  - give the wrong instruction.
#
# Each direction was either "Straight", "Left", or "Right".
#
# You and you friends would like to figure out who made the most mistakes.
# Fortunately everyone wrote down the instructions they gave. Write a function
# ('blame') to figure out who is to blame.
#
# directions = [('map', 'SSSRSSLRLS'), ('jane', 'SSRSLSLSSRLS'), ('jayna',
#                'SRSLSLSRLS'), ('jomo', 'SRSLRRSLSRSLSR')]
#
# >>> print(blame(directions))
# 'jayna'
#
#
#
# Baggage:
# -----------
#
# Next semester you need to move to a new country.  You have lots of useful things
# that you'd like to take with you, but your baggage allowance for the flight is
# very small.  The airline does allow you to pay extra for each bag that you want
# to take.  Anything that you don't pack you will need to buy in the new country.
#
# You need to decide how many extra bags to take, and what you should pack in
# those bags.
#
# >>> weights = {'glasses': 1.0,'plates': 0.7, 'coffee': 1.1,'keyboard': 1.1,
#                'pens': 2.1, 'socks': 0.5, 'undies': 0.3, 'laptop': 2.1,
#                'jersey': 0.5, 'shoes': 0.7, 'jeans': 1.1}
# >>> values = {'glasses': 0.1,'plates': 0.1, 'coffee': 0.1,'keyboard': 0.1,
#               'pens': 1.1, 'socks': 3.1, 'undies': 3.1, 'laptop': 10.1,
#               'jersey': 2.1, 'shoes': 2.1, 'jeans': 2.1}
# >>> baggage_limit = 3
# >>> extra_cost = 10
# >>> print(baggage(weights, values, baggage_limit, extra_cost))
# [['socks', 'undies', 'laptop'], ['jersey', 'shoes', 'jeans']]
#
#


import heapq
import hashlib
from timeit import timeit
from functools import partial


def which_problem(email, seminar, problems):
    email = email.strip().lower()
    assert "@minerva.kgi.edu" in email
    seminar = seminar.strip().lower()
    md5 = hashlib.md5(email + seminar).hexdigest()
    ind = int(md5, 16) % len(problems)
    return problems[ind]


email = 'guy@minerva.kgi.edu'
print(which_problem(email, '12.1', ['directions', 'aquarium', 'baggage']))

# Aquarium:
# -----------
#
# The Cape Town aquarium is designing a new feature and asks you to help. There
# will be fish tanks of different sizes all arranged in a large circle. Each fish
# tank will only contain fish from a single species. If the same fish species are
# stocked in adjacent tanks then they will continually attempt to fight and
# eventually will die of stress.
#
# You will be given a list of fish species, and how much it costs to stock those
# fish per liter of water. You will also be given a list of the tanks and how many
# liters each tank is. Your job is to find which fish should be stocked in which
# tank to achieve minimum cost without incurring any stress on the fish.
#
# Don't forget that the tanks are in a circle, so the beginning and ending tanks
# also mustn't contain the same species either.
#
# >>> tanks = [10, 15, 200, 35, 18, 99, 99, 10]
# >>> fish = [('shark', 12.1), ('marlin', 8.1), ('sole', 9.1)]
# >>> print(aquarium(tanks, fish))
# ['marlin', 'sole', 'marlin', 'sole', 'marlin', 'sole', 'marlin', 'shark']


def djikstra_double_cost_coloring(graph, color_costs):
    frontier = []
    heapq.heappush(frontier, (0, dict(index=0, coloring=[]), ))

    graph_length = len(graph)

    min_total_cost = float("Inf")
    optimal_coloring = None

    while frontier:
        total_cost, current = heapq.heappop(frontier)
        total_cost *= -1

        if total_cost > min_total_cost:
            continue

        index = current['index']
        coloring = current['coloring']
        previous_color = coloring and coloring[-1] or None

        if graph_length == index:
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                optimal_coloring = coloring
            continue

        for color in color_costs:
            if color != previous_color or (graph_length == index + 1 and color != coloring[0]):
                new_total_cost = total_cost + graph[index] * color_costs[color]
                heapq.heappush(frontier, (-1 * new_total_cost,
                                          dict(index=index + 1,
                                               coloring=coloring + [color]), ))

    return min_total_cost, optimal_coloring


def double_sided_bottom_up_aquarium(tanks, fish_costs):
    """
    This is a bottom-up (or rather, middle-out) solution for the aquarium problem.
    I start by building the value for the middle tank (if n is odd) or two middle
    ones (if n is even).
    Then I loop out until I reach the first and last indices of the tanks list,
    where the two sides meet again.

    As spelled out inside, the solution is at worst case O(nk^3),
    and on average should be closer to O(nk^2).

    :param tanks: The sizes of each tank, a list - size of n
    :param fish_costs: A dictionary of fish : cost - size of k
    :return: The minimum cost and assignment that yields it
    """
    # A nested cache from pair of indices (left_index, right_index),
    # to a second cache from fish assignment for those indices to the minimum cost it can incur
    # the cost is the entire cost from those indices to the 'middle'
    cache = {}
    # A nested cache for the paths - the key structure is the same as the above cache,
    # the values are assignments generating these minimum costs
    assignments = {}
    n = len(tanks)
    current_cache = {}
    current_assignments = {}

    # For odd n, fix the 'middle' of the tanks - O(k) runtime
    if 1 == n % 2:
        left_index = right_index = n / 2

        for fish in fish_costs:
            current_assignments[fish, fish] = [fish]
            current_cache[fish, fish] = tanks[left_index] * fish_costs[fish]

    # For even n, fix the 'middle' two entries of the tanks - O(k^2) runtime
    else:
        right_index = n / 2
        left_index = right_index - 1

        for left_fish in fish_costs:
            for right_fish in fish_costs:
                if left_fish != right_fish:
                    current_assignments[left_fish, right_fish] = [left_fish, right_fish]
                    current_cache[left_fish, right_fish] = \
                        tanks[left_index] * fish_costs[left_fish] + \
                        tanks[right_index] * fish_costs[right_fish]

    # Save the results for the 'middle'
    cache[left_index, right_index] = current_cache
    assignments[left_index, right_index] = current_assignments

    # Iterate over all indices remaining - O(n)
    while left_index > 0:
        # The current indices we're finding optimal fish for
        left_index -= 1
        right_index += 1

        # The previous (towards the middle) indices we use to build the current costs
        l = left_index + 1
        r = right_index - 1

        # Take the values for the previous indices, sort them by cost ascending
        previous_cache = cache[(l, r)].items()
        # This could be optimized by keeping the previous cache sorted
        previous_cache.sort(key=lambda x: x[1])

        current_cache = {}
        current_assignments = {}

        # Iterate over all possible fish assignments for the current indices - O(k^2)
        for left_fish in fish_costs:
            for right_fish in fish_costs:
                # If we're back at the edges, make sure we satisfy the circularity constraint
                if left_index == 0 and (left_fish == right_fish):
                    continue

                # Iterate over the results from the previous fish, in ascending cost
                # previous cache is O(k^2), but in the worst case we check 2k + 1 assignments
                # in the best case, 1, and I believe that on expectation, fewer than k k
                for (previous_left_fish, previous_right_fish), previous_cost in previous_cache:
                    if left_fish != previous_left_fish and right_fish != previous_right_fish:
                        current_assignments[left_fish, right_fish] = \
                            [left_fish] + assignments[(l, r)][(previous_left_fish, previous_right_fish)] + [right_fish]

                        current_cache[left_fish, right_fish] = \
                            tanks[left_index] * fish_costs[left_fish] + \
                            tanks[right_index] * fish_costs[right_fish] + \
                            previous_cost

                        break

        cache[left_index, right_index] = current_cache
        assignments[left_index, right_index] = current_assignments

    # Find the optimal solution
    edges = (0, n - 1)
    edges_cache = cache[edges]
    min_start = min(edges_cache, key=edges_cache.get)
    return cache[edges][min_start], assignments[edges][min_start]


def aquarium(tanks, fish):
    fish_costs = {x[0]: x[1] for x in fish}
    # print djikstra_double_cost_coloring(tanks, fish_costs)
    # print
    print timeit(partial(double_sided_bottom_up_aquarium, tanks, fish_costs), number=1000)


def test_aquarium():
    tanks = [10, 15, 200, 35, 18, 99, 99, 10]
    more_tanks = [94, 50, 26, 5, 44, 17, 66, 64, 5, 44,
                  95, 77, 49, 70, 80, 19, 38, 94, 9, 42,
                  94, 62, 47, 56, 32, 45, 78, 52, 93, 55,
                  19, 24, 98, 5, 75, 89, 83, 54, 31, 73,
                  98, 19, 84, 58, 60, 86, 27, 61, 48, 6,
                  38, 12, 17, 7, 53, 50, 46, 45, 76, 52,
                  54, 37, 41, 25, 84, 99, 71, 63, 94, 74,
                  44, 1, 45, 11, 19, 19, 22, 95, 58, 4, 87,
                  2, 81, 83, 56, 99, 69, 47, 94, 81, 57, 38,
                  77, 81, 80, 23, 48, 59, 5, 93]
    fish = [('shark', 12.1), ('marlin', 8.1), ('sole', 9.1)]
    more_fish = [('A', 9.0851815522856825),
                ('B', 3.2187323664433611),
                ('C', 9.053126196179873),
                ('D', 9.3551337309704063),
                ('E', 1.4380480601666867),
                ('F', 2.7222668727185151),
                ('G', 1.4754697364123814),
                ('H', 3.8845813499165738),
                ('I', 7.9875632841325466)]

    aquarium(tanks, fish)
    aquarium(more_tanks, more_fish)


if __name__ == '__main__':
    test_aquarium()
