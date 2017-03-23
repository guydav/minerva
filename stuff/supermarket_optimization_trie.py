import argparse
import os
import time
import pygtrie
import tqdm


DATE_AND_TIME_FORMAT = '%Y_%m_%d_%H_%M_%S'
DEFAULT_MINIMUM_SIZE = 3
DEFAULT_OUTPUT_FILE_TEMPLATE = 'supermarket_optimizer_output_{time}.dat'


class SupermarketOptimizer(object):
    def __init__(self, min_group_size):
        self.min_group_size = min_group_size
        self.occurrence_trie = pygtrie.Trie()

    def process_input(self, input_file, assume_unsorted=False):
        """
        Process an input file into the trie underlying the solution. Take each
         row, parse it into a tuple of items, and insert them if they don't
         exist, or increment their count if they do.
        :param input_file: an open Python file object, or any other iterable
            whose iteration returns strings representing transactions
        :param assume_unsorted: by default False, should we assume each line
            in the input file is unsorted, and hence, sort it
        :return: None; input file contents processed into optimizer
        """
        for line in input_file:
            group = line.strip().split(' ')

            if len(group) < self.min_group_size:
                continue

            if assume_unsorted:
                group = sorted(group)

            if group not in self.occurrence_trie:
                self.occurrence_trie[group] = 1

            else:
                self.occurrence_trie[group] += 1

    def _generate_diffusal_traversal_callback(self, group_set, group_count):
        """
        Generate a callback for a given group (represented as a set) to
        traverse the trie with. Beyond the base case, of always traversing the
        root of the trie, the traversal has two rules:
            - If the group we are diffusing (since it did not have enough
              support does not contain (formally: is not a super set) of the
              current group in the traversal, stop traversin the current branch
            - Otherwise, if the group has a value (we saw it in at least o
        :param group_set:
        :param group_count:
        :return:
        """
        def callback(path_conv, current_group, children,
                     current_group_count=None):
            # Root node - always continue traversing
            if not current_group:
                [None for _ in children]

            # Group being diffused does not contain current group - no need
            # to traverse farther
            if not group_set.issuperset(current_group):
                return

            # Group exists as a node in the trie, meaning it is a group we've
            # seen in the data - increase it by the count of the diffused group
            if current_group_count is not None:
                self.occurrence_trie[current_group] += group_count

            # Continue traversing children of the current group, if any exist
            [None for _ in children]

        return callback

    def _diffuse_group(self, group):
        group_set = set(group)
        self.occurrence_trie.traverse(
            self._generate_diffusal_traversal_callback(
                group_set, self.occurrence_trie[group]))

    def find_supported_groups(self, support_level, output_file):
        """
        TODO: document
        :param support_level:
        :param output_file:
        :return:
        """
        groups_by_length = sorted(self.occurrence_trie.keys(),
                                  key=len,
                                  reverse=True)

        output = []

        for group in tqdm.tqdm(groups_by_length):
            if self.occurrence_trie[group] >= support_level:
                output.append(group)

            else:
                self._diffuse_group(group)

        output_file.writelines([self._format_output(group) for group in output])
        output_file.flush()

    def _format_output(self, group):
        """
        Output a group to the output file, in the format given:
        <group size> <group count> <item 1> <item 2>... <item n>
        :param group: The group to output, a tuple
        :param group_count: The count it appeared in the transactions processed
        :param output_file: The file (or file-like item, supporting a write
            method
        :return:
        """
        output = [str(len(group)), str(self.occurrence_trie[group])]
        output.extend(group)
        return ', '.join(output) + '\r\n'


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
    optimizer = SupermarketOptimizer(args.group_size)
    optimizer.process_input(args.input_file, args.assume_unsorted)
    with open(args.output_file, 'w') as output_file:
        optimizer.find_supported_groups(args.support_level, output_file)

if __name__ == '__main__':
    main()