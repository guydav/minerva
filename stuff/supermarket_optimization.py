import argparse
import collections
import os
import time
import tqdm
import itertools
# import scipy.misc


DATE_AND_TIME_FORMAT = '%Y_%m_%d_%H_%M_%S'
DEFAULT_MINIMUM_SIZE = 3
DEFAULT_OUTPUT_FILE_TEMPLATE = 'supermarket_optimizer_output_{time}.dat'
DEFAULT_COMBINATIONS_THRESHOLD = 10


class SupermarketOptimizer(object):
    def __init__(self, min_group_size):
        self.min_group_size = min_group_size
        self.occurrence_table = collections.defaultdict(int)
        self.diffused_table = collections.defaultdict(int)
        self.groups_by_size = collections.defaultdict(set)

    def process_input(self, input_file, assume_unsorted=False):
        """
        Process the input file, counting how many times each purchsae recurred
        :param input_file: an open Python file object, or any other iterable
            whose iteration returns strings representing transactions
        :param assume_unsorted: by default False, should we assume each line
            in the input file is unsorted, and hence, sort it
        :return: None; input file contents processed into optimizer
        """
        for line in input_file:
            group = map(int, line.strip().split(' '))
            group_size = len(group)

            if group_size < self.min_group_size:
                continue

            if assume_unsorted:
                group = sorted(group)

            group = tuple(group)

            self.occurrence_table[group] += 1
            self.groups_by_size[group_size].add(group)

    def _diffuse_group_by_intersection(self, group):
        """
        I came up with two good ways to 'diffuse' a group which is
        insufficiently supported. This is the first, by intersection. Find the
        intersection of the current, diffused group with each smaller one.
        Eventually, increment all of the intersection groups, counting each one
        only once, even if it was the result of an intersection with multiple
        smaller groups.

        :param group: The group currently being diffused
        :return: None; diffuse the number of times this group has been observed
            to all of its intersection groups
        """
        group_set = set(group)
        intersections = set()

        for smaller_group_size in range(len(group) - 1,
                                        self.min_group_size - 1, -1):

            # Check if from here on out, we are better off using combinations
            # rather than sub-group intersections. Since we only increase the
            # number of sub-groups, making intersections more expensive,
            # we only need to make this decision once
            if self._should_diffuse_by_combinations(len(group),
                                                    smaller_group_size):

                # Increment the subgroups we won't reach by combinations -
                # all those with a larger size than the maximal size we'll find
                # combinations for
                for large_intersection in filter(
                        lambda x: len(x) > smaller_group_size, intersections):
                    self._increment_diffused_group_counts(group,
                                                          large_intersection)

                self._diffuse_group_by_combinations(group, smaller_group_size)
                return

            for smaller_group in self.groups_by_size[smaller_group_size]:
                intersection = group_set.intersection(smaller_group)
                if len(intersection) >= self.min_group_size:
                    intersections.add(tuple(intersection))

        for intersection in intersections:
            self._increment_diffused_group_counts(group, intersection)

    def _diffuse_group_by_combinations(self, group, max_size=None):
        """
        Diffuse a group by incrementing its value to all combinations
        of subgroups thereof, from the minimal group size to the optional
        max_size parameter. The idea was that at some point, there are far
        fewer potential combinations of the current group than subsets that
        have already been observed.

        The issue becomes that we must keep track of every such combination, not
        only ones we have seen before, as we might encounter these combinations
        again later. Therefore, while this idea runs much faster (benchmarked
        it at 10x if not better), it demands too much memory.

        The half-measure I settled on is not using it whenever there are fewer
        combinations to check than potential subsets, but after a hard-coded
        threshold.

        :param group: The group we are currently diffusing
        :param max_size: The size of subgroup to search up to, inclusive
        :return: None; sub-groups all incremented
        """
        if max_size is None:
            max_size = len(group) - 1

        for combination_size in range(self.min_group_size, max_size + 1):
            for comb_group in itertools.combinations(group, combination_size):
                self._increment_diffused_group_counts(group, comb_group)

    def _should_diffuse_by_combinations(self, group_size, sub_group_size=None):
        """
        Estimate if we should diffuse by combinations, rather than by
        intersections of the current group with subgroups thereof.

        Initially I tried that approach, and left in the code - but since it was
        using too much memory, I decided to take a half-measure, and use the
        combination-based approach only after a hard-coded threshold.

        :param group_size: The size of the group currently being diffused
        :param sub_group_size: The maximal size of subgroup to consider -
            relevant when we already dealt with some larger subgroups by
            intersection. This number is inclusive.
        :return: True if we should switch to diffusing by combinations; False
            otherwise
        """
        return group_size <= DEFAULT_COMBINATIONS_THRESHOLD

        # if sub_group_size is None:
        #     sub_group_size = group_size
        #
        # else:
        #     # Since it is inclusive, we increment by one to use with range
        #     sub_group_size += 1
        #
        # smaller_group_count = sum([len(self.groups_by_size[size]) for size in
        #                            range(self.min_group_size, sub_group_size)])
        #
        # combination_count = sum([scipy.misc.comb(group_size, k) for k in
        #                         range(self.min_group_size, sub_group_size)])
        #
        # return combination_count <= smaller_group_count

    def _increment_diffused_group_counts(self, group, sub_group):
        """
        Increment the count of a found sub-group of a given group.

        There are two cases: if this subgroup is heretofore unobserved, we must
        add it to the dictionary of groups by size. Moreover, the original
        group gets to donate both its ground truth observations (from
        the occurrence_table), and the diffused observations it received from
        the diffused_table).

        Otherwise, if a subgroup already exists, we only add the original,
        ground truth occurrences. The reasoning is that if a group exists, it
        received 'diffusions' as groups were being broken down before it. Only
        if it is being created anew does it need to receive these previous
        diffusions, as they were not received.

        :param group: The group being diffused
        :param sub_group: A group it is being diffused into
        :return: None; the group diffused properly
        """
        # This group is new, never observed by itself in the raw data
        if sub_group not in self.occurrence_table:
            self.occurrence_table[sub_group] = 0
            self.groups_by_size[len(sub_group)].add(sub_group)
            self.diffused_table[sub_group] += self.diffused_table[group]

        self.diffused_table[sub_group] += self.occurrence_table[group]

    def find_supported_groups(self, support_level, output_file):
        """
        Find supported groups, by traversing the groups observed from largest
        to smallest, counting for output when sufficient and diffusing when
        necessary.
        :param support_level: The level at or above which we report a group as
            sufficiently supported
        :param output_file: The file (or file-like object, supporting a
            writelines(iterable) method to output into.
        :return: None; output written to output_file
        """
        max_group_size = max(self.groups_by_size.keys())
        output = []

        for group_size in tqdm.trange(max_group_size,
                                      self.min_group_size - 1, -1):
            groups_at_size = self.groups_by_size[group_size]

            for group in tqdm.tqdm(groups_at_size):
                total_group_count = self.occurrence_table[group] + \
                                    self.diffused_table[group]
                if total_group_count >= support_level:
                    output.append((group, total_group_count))

                else:
                    self._diffuse_group_by_intersection(group)

        output_file.writelines([self._format_output(group, count)
                                for group, count in output])
        output_file.flush()

    def _format_output(self, group, count):
        """
        Output a group to the output file, in the format given:
        <group size> <group count> <item 1> <item 2>... <item n>
        :param group: The group to output, a tuple
        :param group_count: The count it appeared in the transactions processed
        :param output_file: The file (or file-like item, supporting a write
            method
        :return:
        """
        output = [str(len(group)), str(count)]
        output.extend(map(str, group))
        return ', '.join(output) + '\r\n'


def generate_output_file_path():
    """
    Generate an default format for an output file, in the current directory,
    using date and timestamp.
    :param support_level:
    :return:
    """
    return os.path.join(os.curdir, DEFAULT_OUTPUT_FILE_TEMPLATE.format(
        time=time.strftime(DATE_AND_TIME_FORMAT)))


def positive_integer_type(value):
    """
    A function used as an argparse type, validating that a given textual input
    is a valid positive integer.
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
    The main function for this module - instantiates an argparse to parse
    input from the command line, and creates and runs a SupermarketOptimizer
    instance
    :return: None
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

    optimizer = SupermarketOptimizer(args.group_size)
    optimizer.process_input(args.input_file, args.assume_unsorted)

    with open(args.output_file, 'w') as output_file:
        optimizer.find_supported_groups(args.support_level, output_file)

if __name__ == '__main__':
    main()
