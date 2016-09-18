from profilehooks import profile
from numpy import random


def left_child(index):
    return 2 * index + 1


def right_child(index):
    return 2 * index + 2


def swap(input_list, a, b):
    """
    Swapping two places in a list extracted to a function
    :param input_list: the list to check in
    :param a: the first index
    :param b: the second index
    :return: No return value; values in the list swapped places
    """
    temp = input_list[a]
    input_list[a] = input_list[b]
    input_list[b] = temp


def max_heapify(input_list, index, heap_size):
    """
    Assumes that the sub-trees of the two children are valid max-heaps, but the root index might not be.
    :param input_list: The list representing the heap
    :param index: The index to max-heapify
    :param heap_size: The size of the heap within the list
    :return: None, with list from the index onwards being a valid max heap
    """
    left = left_child(index)
    right = right_child(index)
    largest = index

    if left < heap_size and input_list[left] > input_list[largest]:
        largest = left

    if right < heap_size and input_list[right] > input_list[largest]:
        largest = right

    if largest != index:
        swap(input_list, largest, index)
        max_heapify(input_list, largest, heap_size)


def build_max_heap(input_list):
    """
    Builds a max-heap from a given list, in place.
    :param input_list: The least to max-heapify
    :return: None, with the list now as a max-heap
    """
    heap_size = len(input_list)
    for index in xrange(heap_size / 2, 0, -1):
        # subtracting 1 from index, as we're actually zero-indexed:
        max_heapify(input_list, index - 1, heap_size)


def heap_sort(input_list):
    build_max_heap(input_list)
    heap_size = len(input_list)

    for index in xrange(heap_size, 1, -1):
        # subtracting 1 from index, as we're actually zero-indexed:
        swap(input_list, 0, index - 1)
        heap_size -= 1
        max_heapify(input_list, 0, heap_size)


def is_sorted(input_list):
    """
    Helper function to test if a list is already sorted
    :param input_list: the list to test
    :return: True if it is already sorted, false otherwise
    """
    return all(input_list[i] <= input_list[i + 1] for i in xrange(len(input_list) - 1))


def main():
    # 3.2
    ints = [i for i in random.random_integers(0, 100, 100)]
    print ints
    heap_sort(ints)
    print ints
    print is_sorted(ints)


if __name__ == '__main__':
    main()