import math
import random
import numpy as np


def radix_sort(input_list):
    # sorted_list = []
    list_max = max(input_list)
    highest_digit = math.floor(math.log10(list_max))
    return recursive_radix(input_list, highest_digit, 0)


def recursive_radix(input_list, highest_digit, prefix):
    buckets = [[] for _ in range(10)]
    power_of_ten = 10 ** highest_digit
    for number in input_list:
        buckets[number // power_of_ten].append(number if power_of_ten == 1 else number % power_of_ten)

    if highest_digit > 0:
        for index in range(len(buckets)):
            if len(buckets[index]) == 1:
                buckets[index][0] += index * power_of_ten

            elif len(buckets[index]) > 1:
                buckets[index] = recursive_radix(buckets[index], highest_digit - 1, index * power_of_ten)

    sorted_segment = []  # can be collapsed into a list comprehension
    for bucket in buckets:
        for number in bucket:
            sorted_segment.append(prefix + number)

    return sorted_segment


def smallest_difference(first_list, second_list):
    first_list = radix_sort(first_list)
    second_list = radix_sort(second_list)

    first_index = 0
    second_index = 0
    min_difference = abs(first_list[first_index] - second_list[second_index])

    last_first = len(first_list) - 1
    last_second = len(second_list) - 1

    while first_index <= last_first and second_index <= last_second:
        current_difference = abs(first_list[first_index] - second_list[second_index])
        if current_difference < min_difference:
            min_difference = current_difference

        if first_index == last_first:
            # won’t overflow since if both done it fails the while
            # TODO: we can stop here once the second list is larger, since difference only grows
            second_index += 1

        elif second_index == last_second:
            # TODO: we can stop here once the first list is larger, since difference only grows
            first_index += 1

        elif first_list[first_index] < second_list[second_index]:
            first_index += 1

        else:  # first_list[first_index] >= second_list[second_index]:
            second_index += 1

    return min_difference


def bottom_up_contiguous_sequence(input):
    n = len(input)
    cache = {(i, i + 1): input[i] for i in range(n)}
    max_sum = max(input)
    max_indices = np.argmax(input)  # TODO: can implement, of course
    max_indices = (max_indices, max_indices + 1)

    for seq_length in range(2, n):
        for i in range(0, n - seq_length):
            indices = (i, i + seq_length)
            value = cache[(i, i + seq_length - 1)] + input[i + seq_length - 1]
            cache[indices] = value
            if value > max_sum:
                        max_sum = value
                        max_indices = indices

    return max_sum, max_indices, input[max_indices[0] : max_indices[1]]


import operator


OPERATOR_DICT = {'*': operator.mul, '/': operator.truediv, '+': operator.add, '-': operator.sub}
OPERATOR_PRECEDENCE = {'*': 1, '/': 1, '+': 0, '-': 0}


def calculator(input_string, operator_precedence=OPERATOR_PRECEDENCE, operator_ops=OPERATOR_DICT):
    operator_stack = []
    number_stack = []
    last_digit = False

    for token in input_string:
        if token.isspace():
            continue

        if token.isdigit():
            if not last_digit:
                last_digit = True
                number_stack.append(int(token))

            else:
                number_stack.append(int(token) + 10 * number_stack.pop())

        else:
            last_digit = False
            if not len(operator_stack):
                operator_stack.append(token)
                continue

            prev_operator = operator_stack[len(operator_stack) - 1]
            if operator_precedence[prev_operator] < operator_precedence[token]:
                operator_stack.append(token)
                continue

            second = number_stack.pop()
            first = number_stack.pop()
            op = operator_stack.pop()
            number_stack.append(operator_ops[op](first, second))
            operator_stack.append(token)

    while len(operator_stack):
        second = number_stack.pop()
        first = number_stack.pop()
        op = operator_stack.pop()
        number_stack.append(operator_ops[op](first, second))

    print(input_string, '=', number_stack[0])
    return number_stack[0]


class BiNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        return '{val} ({left}, {right})'.format(val=self.value,
                                                left= self.left.value if self.left else '',
                                                right=self.right.value if self.right else '')


def tree_to_doubly_linked_list(node, left_parent=BiNode('start'), right_parent=BiNode('end')):
    if node.left:
        left_parent.right = node.left
        tree_to_doubly_linked_list(node.left, left_parent, node)

    else:
        left_parent.right = node
        node.left = left_parent

    if node.right:
        right_parent.left = node.right
        tree_to_doubly_linked_list(node.right, node, right_parent)

    else:
        right_parent.left = node
        node.right = right_parent

    return left_parent


def test_tree_to_dll():
    ll, lr, rl, rr = BiNode(1), BiNode(3), BiNode(5), BiNode(7)

    left_child = BiNode(2, ll, lr)
    right_child = BiNode(6, rl, rr)

    root = BiNode(4, left_child, right_child)

    start = tree_to_doubly_linked_list(root)
    values = [start.value]
    while start.right is not None:
        start = start.right
        values.append(start.value)

    print(' <=> '.join([str(val) for val in values]))


def select_appointments(remaining, selected=None, cache=None):
    if cache is None:
        cache = {}
    if selected is None:
        selected = []

    key = tuple(remaining)
    if key in cache:
        return cache[key]

    # end conditions for recursive algorithm
    if len(remaining) == 0:
        return selected
    if len(remaining) == 1:
        selected.append(remaining[0])
        return selected
    if len(remaining) == 2:
        selected.append(max(remaining))
        cache[key] = selected
        return selected

    first_selected = selected[:]
    first_selected.append(remaining[0])
    first_selected = select_appointments(remaining[2:], first_selected, cache)

    second_selected = selected[:]
    second_selected.append(remaining[1])
    second_selected = select_appointments(remaining[3:], second_selected, cache)
    choice = first_selected if sum(first_selected) > sum(second_selected) else second_selected

    cache[key] = choice
    return choice


def shortest_supersequence(target, input):
    target_locations = {i: [] for i in target}
    for index, value in enumerate(input):
        if value in target_locations:
            target_locations[value].append(index)

    current_indices = {i: 0 for i in target_locations}
    current_locations = {i: target_locations[i][0] for i in target_locations}
    stopping_point = {i: len(target_locations) - 1 for i in target_locations}

    best_indices = min(current_locations.values()), max(current_locations.values())
    best_length = best_indices[1] - best_indices[0] + 1

    # we might be able to optimize by stopping early
    # if we exhaust one list and the cost rises, we should be able to abort
    while current_indices != stopping_point:
        potential_increments = list(filter(lambda key: current_indices[key] != stopping_point[key],
                                current_locations.keys()))
        increment = min([(current_locations[key], key) for key in potential_increments])[1]
        current_indices[increment] += 1
        current_locations[increment] = target_locations[increment][current_indices[increment]]
        proposal_indices = min(current_locations.values()), max(current_locations.values())
        proposal_length = proposal_indices[1] - proposal_indices[0] + 1
        if proposal_length < best_length:
            best_length = proposal_length
            best_indices = proposal_indices

    return best_length, best_indices, input[best_indices[0]: best_indices[1] + 1]


def missing_two(data, threshold=1e-5):
    n = len(data) + 2
    data_sum = sum(data)
    missing_sum = int(n * (n + 1) / 2 - data_sum)
    log_sum = sum(np.log(data))
    full_log_sum = sum([np.log(i) for i in range(1, n + 1)])
    missing_log_sum = full_log_sum - log_sum

    for i in range(1, int(np.ceil(missing_sum / 2))):
        j = missing_sum -i
        if abs(missing_log_sum - np.log(i) - np.log(j)) < threshold:
            return i, j

    return None, None


def test_missing_two(n=100):
    data = list(range(1, n + 1))
    np.random.shuffle(data)
    missing = data[:2]
    print(missing, missing_two(data[2:]))


if __name__ == '__main__':
    # rand_list = [random.randint(1, 999) for _ in range(10)]
    # print(rand_list)
    # print(radix_sort(rand_list))
    # print(smallest_difference([1, 3, 15, 11, 2], [23, 127, 235, 19, 8]))
    # print(bottom_up_contiguous_sequence([2, -8, 3, -2, 4, -10]))
    # calculator('2*3+5/6*3+15')
    # test_tree_to_dll()
    # choice = select_appointments([30, 15, 60, 75, 45, 15, 15, 45])
    # print(sum(choice), choice)
    # print(shortest_supersequence([1, 5, 9], [7, 5, 9, 13, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7]))
    test_missing_two()
