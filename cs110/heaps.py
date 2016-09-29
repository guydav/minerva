from profilehooks import profile
from numpy import random


def parent(index):
    return (index / 2) + (index % 2) - 1


def left_child(index):
    return (2 * index) + 1


def right_child(index):
    return (2 * index) + 2


def swap(heap_list, a, b):
    """
    Swapping two places in a list extracted to a function
    :param heap_list: the list to check in
    :param a: the first index
    :param b: the second index
    :return: No return value; values in the list swapped places
    """
    temp = heap_list[a]
    heap_list[a] = heap_list[b]
    heap_list[b] = temp


def max_heapify(heap_list, index, heap_size):
    """
    Assumes that the sub-trees of the two children are valid max-heaps, but the root index might not be.
    :param heap_list: The list representing the heap
    :param index: The index to max-heapify
    :param heap_size: The size of the heap within the list
    :return: None, with list from the index onwards being a valid max heap
    """
    left = left_child(index)
    right = right_child(index)
    largest = index

    if left < heap_size and heap_list[left] > heap_list[largest]:
        largest = left

    if right < heap_size and heap_list[right] > heap_list[largest]:
        largest = right

    if largest != index:
        swap(heap_list, largest, index)
        max_heapify(heap_list, largest, heap_size)


def build_max_heap(heap_list):
    """
    Builds a max-heap from a given list, in place.
    :param heap_list: The list to max-heapify
    :return: None, with the list now as a max-heap
    """
    heap_size = len(heap_list)
    for index in xrange(heap_size / 2, 0, -1):
        # subtracting 1 from index, as we're actually zero-indexed:
        max_heapify(heap_list, index - 1, heap_size)


def heap_sort(heap_list):
    build_max_heap(heap_list)
    heap_size = len(heap_list)

    for index in xrange(heap_size, 1, -1):
        # subtracting 1 from index, as we're actually zero-indexed:
        swap(heap_list, 0, index - 1)
        heap_size -= 1
        max_heapify(heap_list, 0, heap_size)


def is_sorted(heap_list):
    """
    Helper function to test if a list is already sorted
    :param heap_list: the list to test
    :return: True if it is already sorted, false otherwise
    """
    return all(heap_list[i] <= heap_list[i + 1] for i in xrange(len(heap_list) - 1))


def min_heapify(heap_list, index, heap_size):
    """
    Assumes that the sub-trees of the two children are valid min-heaps, but the root index might not be.
    :param heap_list: The list representing the heap
    :param index: The index to max-heapify
    :param heap_size: The size of the heap within the list
    :return: None, with list from the index onwards being a valid max heap
    """
    left = left_child(index)
    right = right_child(index)
    smallest = index

    if left < heap_size and heap_list[left] < heap_list[smallest]:
        smallest = left

    if right < heap_size and heap_list[right] < heap_list[smallest]:
        smallest = right

    if smallest != index:
        swap(heap_list, smallest, index)
        min_heapify(heap_list, smallest, heap_size)


def heapify(heap_list):
    """
    Builds a min-heap from a given list, in place.
    :param heap_list: The list to min-heapify
    :return: None, with the list now as a min-heap
    """
    heap_size = len(heap_list)
    for index in xrange(heap_size / 2, 0, -1):
        # subtracting 1 from index, as we're actually zero-indexed:
        min_heapify(heap_list, index - 1, heap_size)


def heappop(heap_list):
    """
    Remove the smallest value (top of the min-heap),
    and rebuild the heap by moving the last value to the top and diffusing it down
    :param heap_list: The min-heap to pop from
    :return: The previous smallest item in the heap
    """
    # borrowed from python's implementation - cleanest way to deal with possible empty / almost empty lists
    last_item = heap_list.pop()
    if not heap_list:
        return last_item

    min_key = heap_list[0]
    heap_list[0] = last_item
    min_heapify(heap_list, 0, len(heap_list))
    return min_key


def heap_decrease_key(heap_list, index, key):
    if heap_list[index] < key:
        raise Exception("Current key is smaller than new key, aborting...")

    heap_list[index] = key
    parent_index = parent(index)
    while (index > 0) and (heap_list[parent_index] > heap_list[index]):
        swap(heap_list, parent_index, index)
        index = parent_index
        parent_index = parent(index)


def heappush(heap_list, key):
    """
    Push a new item onto the heap, by originally setting it to a huge value, and then setting
    it to its actual key and diffusing it up the heap
    :param heap_list: The list representing the min-heap to add a key to
    :param key: The key to add to the min-heap
    :return: None, with the new list being inside the key
    """
    heap_list.append(float("inf"))  # a large number
    heap_decrease_key(heap_list, len(heap_list) - 1, key)



def main():
    # 3.2
    # ints = [i for i in random.random_integers(0, 100, 100)]
    # print ints
    # heap_sort(ints)
    # print ints
    # print is_sorted(ints)

    # 4.1
    pass


if __name__ == '__main__':
    main()