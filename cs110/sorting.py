from profilehooks import profile, timecall
from numpy import random


def run_function(func, initial_size, num_outputs):
    return [func(initial_size * 2 ** i) for i in xrange(num_outputs)]


def swap(input_list, a, b):
    temp = input_list[a]
    input_list[a] = input_list[b]
    input_list[b] = temp


def compare_gt(input_list, a, b):
    return input_list[a] > input_list[b]


@profile(sort='calls')
def insertion_sort(input_list):
    list_length = len(input_list)

    for j in xrange(1, list_length):
        for i in xrange(0, j):
            # if input_list[i] > input_list[j]:
            if compare_gt(input_list, i, j):
                input_list.insert(i, input_list.pop(j))
                continue

@profile(sort='calls')
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


@profile(sort='calls')
def selection_sort(input_list):
    list_length = len(input_list)

    for i in xrange(list_length):
        min_index = i
        for j in xrange(i+1, list_length):
            if compare_gt(input_list, min_index, j):
                min_index = j

        input_list.insert(i, input_list.pop(min_index))


def main():
    # 1.1
    # run_function(lambda x: [0] * x, 2, 20)

    # 1.2
    ints = [i for i in random.random_integers(0, 100, 100)]
    print ints

    for sort_func in [insertion_sort, bubble_sort, selection_sort]:
        ints_copy = ints[:]
        sort_func(ints_copy)
        print sort_func.func_name
        print ints_copy


if __name__ == '__main__':
    main()
