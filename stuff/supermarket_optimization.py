import argparse
import heapq
import os
import time
import itertools
import tqdm


DATE_AND_TIME_FORMAT = '%Y_%m_%d_%H_%M_%S'
DEFAULT_MINIMUM_SIZE = 3
DEFAULT_OUTPUT_FILE_TEMPLATE = 'supermarket_optimizer_output_{time}.dat'


class MinHeap(object):
    """
    A lightweight OOP wrapper of Python's built-in heapq
    """
    def __init__(self, data=None):
        if data is not None:
            self.heap = heapq.heapify(data)

        else:
            self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    def push_pop(self, item):
        return heapq.heappushpop(self.heap, item)

    def n_smallest(self, n):
        return heapq.nsmallest(n, self.heap)

    def n_largest(self, n):
        return heapq.nsmallest(n, self.heap)

    def peek(self):
        return self.heap[0]

    def heapify(self):
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str(self.heap)

    def __getitem__(self, item):
        return self.heap.__getitem__(item)


class SupermarketOptimizer(object):
    def __init__(self):
        self.occurrence_table = {}
        self.heap = MinHeap()

    def _heap_push(self, line_tuple):
        self.heap.push((-1 * len(line_tuple), line_tuple))

    def _heap_pop(self):
        return self.heap.pop()[1]

    def _heap_pop_generator(self):
        while len(self.heap):
            yield self._heap_pop()

    def process_input(self, input_file, assume_unsorted=False):
        """
        TODO: document
        :param input_file: an open Python file object, or any other iterable
            whose iteration returns strings representing transactions
        :param assume_unsorted: by default False, should we assume each line
            in the input file is unsorted, and hence, sort it
        :return: None; input file contents processed into optimizer
        """
        for line in input_file:
            group = line.strip().split(' ')

            if assume_unsorted:
                group = sorted(group)

            group = tuple(group)

            if group not in self.occurrence_table:
                self.occurrence_table[group] = 1
                self._heap_push(group)

            else:
                self.occurrence_table[group] += 1

    def _output(self, group, group_count, output_file):
        """
        TODO: document
        :param group:
        :param group_count:
        :param output_file:
        :return:
        """
        output = [str(len(group)), str(group_count)]
        output.extend(group)
        output_file.write(', '.join(output) + '\r\n')

    def _break_down_group(self, group, group_count, min_group_size):
        """
        TODO: document
        :param group:
        :param group_count:
        :param min_group_size:
        :return:
        """
        for subgroup_size in range(len(group) - 1, min_group_size - 1, -1):
            for subgroup in itertools.combinations(group, subgroup_size):
                if subgroup in self.occurrence_table:
                    self.occurrence_table[subgroup] += group_count

                    remainder_group = tuple(set(group).difference(subgroup))
                    if len(remainder_group) < min_group_size:
                        return

                    if remainder_group in self.occurrence_table:
                        self.occurrence_table[remainder_group] += group_count
                    else:
                        self._break_down_group(remainder_group, group_count,
                                               min_group_size)

    def find_supported_groups(self, support_level, min_group_size, output_file):
        """
        TODO: document
        :param support_level:
        :param min_group_size:
        :param output_file:
        :return:
        """
        for group in tqdm.tqdm(self._heap_pop_generator(),
                               total=len(self.heap)):
            # Since we are popping from a max-heap, once we reach a group
            # below the minimum size, we know we are done
            if len(group) < min_group_size:
                output_file.flush()
                return

            group_count = self.occurrence_table[group]
            if group_count >= support_level:
                self._output(group, group_count, output_file)

            else:
                self._break_down_group(group, group_count, min_group_size)

        output_file.flush()


def generate_output_file_path():
    """
    TODO: document
    :param support_level:
    :return:
    """
    return os.path.join(os.curdir, DEFAULT_OUTPUT_FILE_TEMPLATE.format(
        time=time.strftime(DATE_AND_TIME_FORMAT)))


def positive_integer_type(value):
    """
    TODO: document
    :param value:
    :return:
    """
    try:
        int_value = int(value)

    except ValueError:
        raise argparse.ArgumentTypeError(
            'Must be a valid integer, received {value}'.format(value=value))

    if 0 >= int_value:
        raise argparse.ArgumentTypeError(
            'Must be a positive integer, received {value}'.format(value=value))

    return int_value


def main():
    """
    TODO: document
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True,
                        type=argparse.FileType('r'),
                        help='Input file - the transaction database')
    parser.add_argument('-o', '--output-file', required=False,
                        default=generate_output_file_path(),
                        help='Output file path - autogenerated if not provided')
    parser.add_argument('-s', '--support-level', required=True,
                        type=positive_integer_type,
                        help='The requisite support level - a positive integer')
    parser.add_argument('-u', '--assume-unsorted', required=False,
                        action='store_true',
                        help='Assume each line in the input file is unsorted')
    parser.add_argument('-g', '--group-size', required=False,
                        default=DEFAULT_MINIMUM_SIZE,
                        help='Override the default minimum group size {size}'
                            .format(size=DEFAULT_MINIMUM_SIZE))

    args = parser.parse_args()
    optimizer = SupermarketOptimizer()
    optimizer.process_input(args.input_file, args.assume_unsorted)
    with open(args.output_file, 'w') as output_file:
        optimizer.find_supported_groups(args.support_level, args.group_size,
                                        output_file)

if __name__ == '__main__':
    main()