from profilehooks import profile
import random


def swap(input_list, a, b):
    temp = input_list[a]
    input_list[a] = input_list[b]
    input_list[b] = temp


def is_sorted(input_list):
    """
    Helper function to test if a list is already sorted
    :param input_list: the list to test
    :return: True if it is already sorted, false otherwise
    """
    return all(input_list[i] <= input_list[i + 1] for i in xrange(len(input_list) - 1))


def clean_quickselect(input_list, k, start=0, end=None):
    if not end:
        end = len(input_list)

    if k > end:
        raise ValueError('k > end')

    pivot = end - 1

    if start == pivot:
        return input_list[start]

    i = start - 1
    for j in xrange(start, pivot):
        if input_list[j] < input_list[pivot]:
            i += 1
            swap(input_list, i, j)

    i += 1
    swap(input_list, i, pivot)

    if i == k:
        return input_list[i]

    if i > k:
        return clean_quickselect(input_list, k, start, i)

    return clean_quickselect(input_list, k, i + 1, end)

if __name__ == '__main__':
    numbers = range(1000)
    random.shuffle(numbers)

    for k in xrange(1000):
        qs = clean_quickselect(numbers, k)
        if qs != k:
            print k, qs





@profile(immediate=True)
def optimized_quicksort(input_list, start=0, end=None):
    if is_sorted(input_list) or len(input_list) == 1:
        return

    if end is None:
        end = len(input_list)

    frontier_quicksort(input_list, start, end)


def frontier_quicksort(input_list, initial_start, initial_end):
    frontier = [(initial_start, initial_end)]

    while frontier:
        start, end = frontier.pop()

        pivot = end - 1

        if start >= pivot:
            continue

        if start + 1 == pivot:
            if input_list[start] > input_list[pivot]:
                temp = input_list[start]
                input_list[start] = input_list[pivot]
                input_list[pivot] = temp

            continue

        i = start - 1
        for j in xrange(start, pivot):
            if input_list[j] < input_list[pivot]:
                i += 1
                temp = input_list[i]
                input_list[i] = input_list[j]
                input_list[j] = temp

        i += 1
        # swap(input_list, i, pivot)
        temp = input_list[i]
        input_list[i] = input_list[pivot]
        input_list[pivot] = temp

        # clean_quicksort(input_list, start, i)
        # clean_quicksort(input_list, i + 1, end)
        frontier.append((start, i))
        frontier.append((i + 1, end))