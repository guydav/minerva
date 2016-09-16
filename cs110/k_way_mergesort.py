from profilehooks import profile
from numpy import random


DEBUG = False


def compare_gt(input_list, a, b):
    """
    Comparison of two places in a list extracted to a function.
    Used in order to profile how many times this comparison happens
    :param input_list: the list to check in
    :param a: the first index
    :param b: the second index
    :return: True if the item at a is larger than the item at b, false otherwise
    """
    return input_list[a] > input_list[b]


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


def is_sorted(input_list):
    """
    Helper function to test if a list is already sorted
    :param input_list: the list to test
    :return: True if it is already sorted, false otherwise
    """
    return all(input_list[i] <= input_list[i + 1] for i in xrange(len(input_list) - 1))

@profile(sort='calls', immediate=True)
def insertion_sort(input_list, start=0, end=None):
    """
    Insertion sort, implemnted from pseudo code by Cormen et al. (CLRS). Assumes the list isn't already sorted.
    :param input_list: the list to sort
    :param start: the index to start from, defaults to 0
    :param end: the index to end on, defaults to the length of the list, exclusive (Python list convention)
    :return: None, with the list sorted in-place
    """
    list_length = len(input_list)

    if end is None:
        end = list_length

    # start+1 as a list of length 1 is always sorted
    for i in xrange(start+1, end):
        current_value = input_list[i]
        j = i - 1

        while (j >= start) and (input_list[j] > current_value):
            input_list[j + 1] = input_list[j]
            j -= 1

        input_list[j + 1] = current_value


@profile(sort='calls', immediate=True)
def k_way_merge_sort(input_list, k=3, start=0, end=None, insertion_sort_threshold=None):
    """
    Fully flexible k-way merge sort. Default value of k=3 makes it a three-way
    merge sort unless a different parameter is passed in.
    :param input_list: The list to sort, assumes it is not already sorted.
    :param k: The number of sub-arrays to sort and merge through every time.
            k > len(input_list) will be treated as len(input_list), k < 1 will be treated as k = 1,
            which is actually insertion sort, so it will call insertion sort directly
    :param start: the start index of the subarray currently being processed
            defaults to 0, to allow ignoring the parameter when calling on the whole array
    :param end: the end index of the subarray currently being processed, exclusive (Python list convention)
            defaults to the length of the list to allow ignoring when calling on the whole array
    :return: None, with the list sorted in-place
    """
    list_length = len(input_list)

    if k > list_length:
        k = list_length

    if end is None:
        end = list_length

    # if k <= 1, we treat it as k = 1, which is insertion sort, so we just call that instead
    # Additionally, checks the insertion sort threshold, where below a certain list length we just use insertion sort
    if (k <= 1) or (insertion_sort_threshold and end - start <= insertion_sort_threshold):
        return insertion_sort(input_list, start, end)

    # If the current subarray is smaller than k, just merge it directly, otherwise split it
    subarray_indices = generate_subarrays(start, end, k)
    if subarray_indices:
        for (subarray_start, subarray_end) in subarray_indices:
            k_way_merge_sort(input_list, k, subarray_start, subarray_end)

    merge(input_list, start, end, subarray_indices)


def generate_subarrays(start, end, k):
    """
    Generate subarray indices for given start index, end index, and k parts.
    Attempts to generate subarrays of as equal length as possible.
    Once it finds a size that evenly divides the remaining part into the remaining number of subarrays,
    stops dividing each time to find the current size, and uses the stable one.

    :param start: the start index of the the current subarray to divide
    :param end: the end index of the the current subarray to divide, exclusive (Python list convention)
    :param k: the number of subarrays to divide to.
    :return: If k < end - start, divides into k subarrays of as equal length as possible
            Otherwise, returns None
    """
    if DEBUG:
        print 'subarrays called, start={start}, end={end}, k={k}'.format(start=start, end=end, k=k)

    if not (end - start > k):
        return None

    subarray_indices = []
    stable_size = None

    for i in xrange(k):
        if stable_size:
            current_end = start + stable_size
            subarray_indices.append((start, current_end))

        else:
            current_subarray_size = (end - start) / (k - i)
            mod = ((end - start) % (k - i))
            if 0 == mod:
                stable_size = current_subarray_size

            current_end = start + current_subarray_size
            subarray_indices.append((start, current_end))

        start = current_end

    if DEBUG:
        print subarray_indices

    return subarray_indices


def merge(input_list, start, end, subarray_indices):
    """
    Merge the subarrays in the section between start and end. Deals with several edge cases:
    - If there's one element, do nothing (one element is always sorted)
    - If there are two elements, swap them if necessary
    - If all "subarrays" are all of length 1, treat it as insertion sort (and call it instead)
    - Otherwise, actually merge the subarrays
    :param input_list: The list to merge in
    :param start: The index the subarrays start in
    :param end: The index the subarrays end in, exclusive (Python list convention)
    :param subarray_indices: The indices of the different subarrays in the interval [start, end),
                            each assumed to be sorted.
    :return: None, with the section of input_list between start and end sorted
    """
    if DEBUG:
        print 'merge called, start={start}, end={end}, subarrays={subarray_indices}'.format(
            start=start, end=end, subarray_indices=subarray_indices)

    # If there's only one element, there is nothing to do
    if start + 1 == end:
        return

    # If there are two elements, they might need to be swapped
    if start + 2 == end:
        if compare_gt(input_list, start, end - 1):
            swap(input_list, start, end - 1)

    # If there are no subarray indices, we're merging "lists" of length 1, which is equivalent to insertion sort:
    if subarray_indices is None:
        insertion_sort(input_list, start, end)

    # the more complicated case, of merging actual sub-lists
    else:
        sub_lists = [input_list[start_index : end_index] for (start_index, end_index) in subarray_indices]

        for index in xrange(start, end):
            min_sub_list_index = 0
            for current_sub_list_index in xrange(1, len(sub_lists)):
                if sub_lists[current_sub_list_index][0] < sub_lists[min_sub_list_index][0]:
                    min_sub_list_index = current_sub_list_index

            input_list[index] = sub_lists[min_sub_list_index].pop(0)
            if not sub_lists[min_sub_list_index]:
                sub_lists.pop(min_sub_list_index)

    if DEBUG:
        print input_list


def main():
    # Stress / random testing code:
    # for _ in xrange(100):
    #     if 0 == (i % 10):
    #         print i
    #
    #     for k in xrange(2, 21):
    #         ints = [i for i in random.random_integers(0, 1000, 1000)]
    #         ints_copy = ints[:]
    #         k_way_merge_sort(ints, k=k)
    #
    #         if not is_sorted(ints):
    #             print ints_copy
    #             print ints
    #             print k

    # More exhaustion testing, this time of all possible values of k
    # for k in xrange(1000):
    #     ints = [i for i in random.random_integers(0, 1000, 1000)]
    #     ints_copy = ints[:]
    #     k_way_merge_sort(ints, k=k)
    #
    #     if not is_sorted(ints):
    #         print k
    #         print ints_copy
    #         print ints

    # More exhaustion testing, this time for values of the insertion sort threshold
    # for threshold in xrange(2, 100):
    #     ints = [i for i in random.random_integers(0, 1000, 1000)]
    #     ints_copy = ints[:]
    #     k_way_merge_sort(ints, insertion_sort_threshold=threshold)
    #
    #     if not is_sorted(ints):
    #         print threshold
    #         print ints_copy
    #         print ints
    ints = [i for i in random.random_integers(0, 10000, 10000)]
    k_way_merge_sort(ints, k=16, insertion_sort_threshold=16)


if __name__ == '__main__':
    main()