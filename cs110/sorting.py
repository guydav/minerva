from profilehooks import profile
from numpy import random


def run_function(func, initial_size, num_outputs):
    return [func(initial_size * 2 ** i) for i in xrange(num_outputs)]


def compare_gt(input_list, a, b):
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


@profile(sort='calls', immediate=True)
def insertion_sort(input_list):
    list_length = len(input_list)

    for j in xrange(1, list_length):
        for i in xrange(0, j):
            # if input_list[i] > input_list[j]:
            if compare_gt(input_list, i, j):
                input_list.insert(i, input_list.pop(j))
                continue


@profile(sort='calls', immediate=True)
def shell_sort(input_list, step_size=None):
    list_length = len(input_list)

    if not step_size or step_size > (list_length * 0.5):
        step_size = int(list_length * 0.5)

    while step_size > 0:
        for i in xrange(step_size):
            # select the current sub-list to sort
            current_sub_list = input_list[i::step_size]

            # sort it
            insertion_sort(current_sub_list)

            # insert the elements back into the original list
            input_list[i::step_size] = current_sub_list

        if step_size == 1:
            step_size = 0

        else:
            step_size = int(step_size * 0.5)

@profile(sort='calls', immediate=True)
def bubble_sort(input_list):
    done = False
    max_index = len(input_list) - 1

    while not done:
        done = True
        for i in xrange(max_index):
            # if input_list[i] > input_list[i+1]:
            if compare_gt(input_list, i, i+1):
                swap(input_list, i, i+1)
                done = False


@profile(sort='calls', immediate=True)
def selection_sort(input_list):
    list_length = len(input_list)

    for i in xrange(list_length):
        min_index = i
        for j in xrange(i+1, list_length):
            if compare_gt(input_list, min_index, j):
                min_index = j

        input_list.insert(i, input_list.pop(min_index))


@profile(sort='calls', immediate=True)
def merge_sort(input_list, start=0, end=None):
    if end is None:
        end = len(input_list)

    # print 'merge sort called, start={start}, end={end}'.format(start=start, end=end)

    # Adding two as it's an open-ended interval
    if start + 2 < end:
        mid = start + (end - start) / 2
        merge_sort(input_list, start, mid)
        merge_sort(input_list, mid, end)
    else:
        mid = None

    merge(input_list, start, mid, end)


"""
Merges the list from start to end, assumes the sublists
[start, mid) and [mid, end) are sorted relative to each other

# if left_index == mid:
#     new_sub_list.append(input_list[right_index])
#     right_index += 1
#
# elif right_index == end:
#     new_sub_list.append(input_list[left_index])
#     left_index += 1
#
# elif compare_gt(input_list, left_index, right_index):
#     new_sub_list.append(input_list[right_index])
#     right_index += 1
#
# else:
#     new_sub_list.append(input_list[left_index])
#     left_index += 1
"""
def merge(input_list, start, mid, end):
    # print 'merge called, start={start}, mid={mid}, end={end}'.format(start=start, mid=mid, end=end)

    if start + 1 == end:
        return

    # the case where start + 1 == end, we might have to swap
    if mid is None:
        if compare_gt(input_list, start, end - 1):
            swap(input_list, start, end - 1)

    # the more complicated case, of merging two sub-lists
    else:
        new_sub_list = []
        left_index = start
        right_index = mid
        while (left_index < mid) or (right_index < end):
            if (left_index == mid) or (right_index != end and compare_gt(input_list, left_index, right_index)):
                new_sub_list.append(input_list[right_index])
                right_index += 1

            else:
                new_sub_list.append(input_list[left_index])
                left_index += 1

        input_list[start:end] = new_sub_list

    # print input_list

def main():
    # 1.1
    # run_function(lambda x: [0] * x, 2, 20)

    # 1.2
    # ints = [i for i in random.random_integers(0, 10000, 10000)]
    # # print ints
    #
    # for sort_func in [insertion_sort, shell_sort, bubble_sort, selection_sort]:
    #     ints_copy = ints[:]
    #     sort_func(ints_copy)
    #     print sort_func.func_name
    #     # print ints_copy

    # 2.1
    ints = [i for i in random.random_integers(0, 1000, 1000)]
    # print ints
    # merge_sort(ints)
    # print ints
    # print all(ints[i] <= ints[i + 1] for i in xrange(len(ints) - 1))

    for sort_func in [insertion_sort, selection_sort, merge_sort]:
        ints_copy = ints[:]
        sort_func(ints_copy)
        print sort_func.func_name, all(ints_copy[i] <= ints_copy[i + 1] for i in xrange(len(ints_copy) - 1))


if __name__ == '__main__':
    main()
