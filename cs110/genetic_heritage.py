#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tabulate
import numpy
from collections import defaultdict
from itertools import permutations


# The gene data from the assignment
RAW_GENE_DATA = (
    (0, 'TCGCCAATATTATATTTTTGAAGGCGTAGCTAATGTGGATACTATGTAAGTCGCAAGCTCTGCCAAACAGGGCTAATGAACAAACACTATAATGAGGAC'),
    (1, 'TCGCAATATATCTTTTTGGAGAGCGTAGTATGTGGATATATCTCAAGTCGCAAGCTCTGCCGAAACAGGGCATGATTAAGGAATTACTACAATGAGGAAA'),
    (2, 'TGCAATATAACTTTTTAGGAGCATAGTAATGTCGGATATATTTTAAGTCGCAAGCGCCGAAACAGGGCACAATGAAGAAACACTATAAAGAGGAAC'),
    (3, 'TGCGCATATTTTTTTCTTAGTGTAGCGTAGTATGTGGAGATTCTCAAGTCGGAAGCTGCCTTAAACAGCGCATGATCAGGATTACTACAATAGAAGGAAA'),
    (4, 'TGCGCAATATATCTTTTTTAGAGAGCGTAGTATTGGTATATTCCCAAGTCGGAAGCTGCTTAAATCACCATGATACATGGAATTACTACAATAGAGGAAA'),
    (5, 'TCGCAATATATCTTTTTGAGAGCGTAGTAATGTGGATATATCTAAGTCGCAAGCTCTGCCGAAACAGGGCATAATGAAGAAACACTATAATGAGGAAC'),
    (6, 'TGCGCAATATATCTTTTTTAGGAGAGCGTAGTATGTGGATATTCTCAAGTCGGAAGCTGCCTTAAACAGCGCATGATAAGGAATTACTACAATAGAGGAAA'))

# The global cache used by the caching decotrator
CACHE = defaultdict(dict)

# Markers used for the tree-printing function
CHILD_ON_LEFT = True
CHILD_ON_RIGHT = False

# Keys for the levenshtein distance editing decisions dictionaries
DEL = 'deletions'
INS = 'insertions'
SUB = 'substitutions'


def cache_decorator(ignore_order=False):
    """
    A decorator supporting caching of a function with any number of arguments
    :param ignore_order: Is the function argument-order sensitive, or not.
    :return: The same function, wrapped to cache
    """

    def cache_function(function):
        global CACHE

        function_cache = CACHE[function]

        def cached_function(*args, **kwargs):
            key = list(args) + kwargs.items()

            if ignore_order:
                key.sort()

            key = tuple(key)

            if key in function_cache:
                return function_cache[key]

            result = function(*args, **kwargs)
            function_cache[key] = result

            return result

        return cached_function

    return cache_function


@cache_decorator(ignore_order=True)
def bottom_up_longest_subsequence(x, y):
    """
    Bottom-up implementation of the longest subsequence algorithm, adapted
    from Cormen et al. (2009). This implementation is adapted to directly
    return the longest subsequence length, rather than the table.
    :param x: The first string to compare
    :param y: The second string to compare
    :return: The length of the longest mutual subsequence between x and y
    """
    m = len(x)
    n = len(y)

    cache = numpy.zeros((m + 1, n + 1))

    for i in xrange(1, m + 1):
        for j in xrange(1, n + 1):
            up = cache[i - 1][j]
            left = cache[i][j - 1]

            if x[i - 1] == y[j - 1]:
                cache[i][j] = cache[i - 1][j - 1] + 1

            elif up >= left:
                cache[i][j] = up

            else:  # up < left
                cache[i][j] = left

    return cache[m][n]


@cache_decorator()
def gene_resemblance(row_index, col_index, genes, resemblance_func,
                     proportion=False, default=None):
    """
    A wrapper to handle resemblance comparison of two genes. Passing
    in the indices and list itself to avoid a string comparison between
    the same row and column, and allow for whomever is calling this
    function to perform a full iteration over rows x cols.
    :param row_index: The first (major) index in the genes list to compare
    :param col_index: The second (minor) index in the genes list to compare
    :param genes: The list of genes, as strings, to compare
    :param resemblance_func: The function used to generate a numerical comparison
        between the two genes. Must accept exactly two parameters, the two strings.
    :param proportion: Whether or not the proportion, relative to the length of the
        row (major) gene should be computed. If True, the output of the resemblance
        function is divided by the length of the row gene
    :param default: A default value to return when a gene is compared with itself.
        Used to allow returning 0 for the edit distance between two genes. Defaults
        to the length of the gene (for the longest subsequence case)
    :return: A comparison between the genes:
        - If the same indices, the default value, or the length of the longest gene
        - If different indices, the output of the resemblance function
        - If proportion is True, the output or default value is divided by
            the length of the row gene.
    """
    row_length = len(genes[row_index])
    result = row_length
    if default is not None:
        result = default

    if row_index != col_index:
        result = resemblance_func(genes[row_index], genes[col_index])

    if proportion:
        result = float(result) / row_length

    return result


def print_table(table, table_format='fancy_grid', gene_indices=range(7)):
    """
    Print a table, using tabulate.
    :param table: The table of values to be printed - assumed to include neither row
        nor column headings
    :param table_format: A format for the table - defaults to tabulate's fancy grid,
        used to also allow to generate LaTeX tables directly
    :param gene_indices: The headings to be used for both rows and columns
    :return: None; table printed to stdout
    """
    table_with_row_headings = [[gene_indices[i]] + table[i] for i in xrange(len(gene_indices))]
    header = [""]
    header.extend([str(i) for i in gene_indices])
    print tabulate.tabulate(table_with_row_headings, header, tablefmt=table_format)


def add_threshold_marker(value, thresholds, larger=True):
    """
    Add a marker if the value is below or above a certain threshold
    :param value: The current value to annotate
    :param thresholds: A list of tuples, each specifying a threshold and a marker for it
    :param larger: Whether the markers should be applied if the value is larger than the
        threshold, or smaller than it
    :return: the value, annotated if it was necessary
    """
    if thresholds is None:
        return value

    marker = ''
    for threshold_value, threshold_marker in thresholds:
        if (larger and value >= threshold_value) or (not larger and value <= threshold_value):
            marker = threshold_marker
            break

    return '{value:.3f}{marker}'.format(value=value, marker=marker)


def generate_similarity_table(gene_indices, genes,
                              resemblance_func=bottom_up_longest_subsequence,
                              proportion=False, default=None, thresholds=None,
                              threshold_larger=True):
    """
    A wrapper function replacing several other functions I had before,
    generating a similarity table between the genes supplied
    :param gene_indices: The list of indices of genes
    :param genes: The list of genes themselves
    :param resemblance_func: The metric used to calculate resemblance between
        two genes - defaults to `bottom_up_longest_subsequence`, levenshtein
        distance is also implemented
    :param proportion: True if the table should output proportions (relative
        to the length of the row gene), False if it should output raw
        resemblance metrics
    :param default: A default value to be passed to `gene_resemblance`
    :param thresholds: If exist, a set of thresholds to be used to annotate
        the values output; see `add_threshold_marker`
    :param threshold_larger: A flag passed to `add_threshold_marker`
    :return: The generated table, without and row or column headers
    """
    table = [[add_threshold_marker(gene_resemblance(
                 row_gene, col_gene, genes, resemblance_func, proportion, default), thresholds, threshold_larger)
              for col_gene in gene_indices]
             for row_gene in gene_indices]

    return table


def greedy_phylogeny_inference(gene_indices, genes,
                               resemblance_func=bottom_up_longest_subsequence,
                               proportion=True, default=None):
    """
    Attempt to infer relationships greedily - start from the value with the highest
    average resemblance to the rest, pick the two most-resembling genes as its children,
    and repeating this process to pick the grand-children.

    Note: this assumes the highest value is best, as it was implemented with the longest
     subsequence in mind. If using with levenshtein distance, multiply values by -1.
    :param gene_indices: The indices of the different genes
    :param genes: The gene strings themselves
    :param resemblance_func: The resemblance metric to use - defaults to the
        `bottom_up_longest_subsequence`, but the levenshtein distance is also
        implemented
    :param proportion: True if proportions should be used, False if raw values
    :param default: A parameter passed to `generate_similarity_table`
    :return:
    """
    table = generate_similarity_table(gene_indices, genes, resemblance_func, proportion, default)

    # Although it's not mathematically correct, since all numbers are positive,
    # it doesn't harm to average in the 1.0 for the self-self proportion
    used = set()
    indices_set = set(gene_indices)

    averages = [(numpy.average(table[index]), index) for index in gene_indices]
    start = max(averages)[1]
    used.add(start)
    queue = [start]
    results = [start]

    while queue and used != indices_set:
        current = queue.pop(0)
        children = [child[1] for child in
                    sorted([(table[current][i], i) for i in gene_indices if i not in used], reverse=True)[:2]]
        used.update(children)
        results.extend(children)
        queue.extend(children)

    print_tree(results)
    return results


def left_child(i):
    """
    Return the left child of a node in a list-backed tree
    :param i: the current index
    :return: The index of the left child
    """
    return 2 * i + 1


def right_child(i):
    """
    Return the right child of a node in a list-backed tree
    :param i: the current index
    :return: the index of the right child
    """
    return 2 * i + 2


def print_tree(tree_list, current=0, depth=0, is_right=None):
    """
    A hack of a printing function to print out a tree repesented in a list
    :param tree_list: A list representing a tree, assuming that for a given
        index, 2 * index + 1 / + 2 are its left and right children
    :param current: The current index to print, defaults to 0 (the root)
    :param depth: The current depth to print at
    :param is_right: Are we on the right or left
    :return:
    """
    if tree_list is None:
        return

    if depth > 10:
        raise ValueError("Unable to comply, building in progress")

    left_index = left_child(current)
    right_index = right_child(current)
    length = len(tree_list)

    if right_index < length:
        print_tree(tree_list, right_index, depth + 1, CHILD_ON_RIGHT)

    if is_right is None:
        symbol = '--'
    elif is_right == CHILD_ON_RIGHT:
        symbol = '/'
    else:
        symbol = '\\'

    print '   ' * depth + ' {symbol} {data}'.format(symbol=symbol, data=tree_list[current])

    if left_index < length:
        print_tree(tree_list, left_index, depth + 1, CHILD_ON_LEFT)


@cache_decorator()
def levenshtein_distance_with_tracking(source, target):
    """
    Adapted from https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix
    This is an implementation of the Wagner–Fischer algorithm, but tracking which "editing decision"
    is made at each point in time, in order to later provide estimates of how often each
    editing decision was taken.

    It is interesting to note that aside of the modifications I wrote in while tracking (which
    could be reconstructed from the transition table, b, in the CLRS implementation of LCS),
    the Levenshtein distance is almost the same as the difference in lengths between the
    source string and the longest common subsequence it has with the target string.
    :param source: the source string
    :param target: the target string
    :return: The minimal number of edits required to arrive form source to target,
        and the dictionary of editing decisions rquired to arrive at that point
    """
    m = len(source)
    n = len(target)
    edits = numpy.zeros((m + 1, n + 1))
    decisions = {(0, 0): {DEL: 0, INS: 0, SUB: 0}}

    # source strings can be reached by dropping all characters
    for i in xrange(1, m + 1):
        edits[i, 0] = i
        decisions[i, 0] = {DEL: i, INS: 0, SUB: 0}

    # target strings can be reached by adding all necessary characters:
    for j in xrange(1, n + 1):
        edits[0, j] = j
        decisions[0, j] = {DEL: 0, INS: j, SUB: 0}

    for j in xrange(1, n + 1):
        for i in xrange(1, m + 1):
            current_sub_cost = source[i - 1] != target[j - 1]

            deletion_cost = edits[i - 1, j] + 1
            insertion_cost = edits[i, j - 1] + 1
            substitution_cost = edits[i - 1, j - 1] + current_sub_cost

            min_cost = min(deletion_cost, insertion_cost, substitution_cost)
            edits[i, j] = min_cost

            if min_cost == deletion_cost:
                decision = decisions[i - 1, j].copy()
                decision[DEL] += 1

            elif min_cost == insertion_cost:
                decision = decisions[i, j - 1].copy()
                decision[INS] += 1

            else:  # min_cost == substitution_cost
                decision = decisions[i - 1, j - 1].copy()
                decision[SUB] += current_sub_cost

            decisions[i, j] = decision

    return edits[m, n], decisions[m, n]


@cache_decorator(ignore_order=True)
def levenshtein_distance(source, target):
    """
    A wrapper for the `levenshtein_distance_with_tracking` function,
    to provide only the editing distance. Since it's symmetric, we
    can use the order-ignoring cache
    :param source: The source string
    :param target: The target string
    :return: The number of edits required to go source -> target
    """
    return levenshtein_distance_with_tracking(source, target)[0]


def levenshtein_probability_inferences(inferred_phylogeny, genes):
    """
    An attempt to infer the probabilities of different editing decisions - insertion,
    deletion, and substitution - form the inferred phylogeny
    :param inferred_phylogeny: the phylogeny inferred, either greedly or exhaustingly,
        as a list-backed binary tree
    :param genes: The gene strings themselves
    :return: None; probability inferences for the different changes printed to stdout
    """
    queue = [inferred_phylogeny[0]]
    total_start_length = 0
    total_edits = 0
    total_decisions = {DEL: 0, INS: 0, SUB: 0}

    results_length = len(inferred_phylogeny)

    while queue:
        current = queue.pop(0)
        current_gene = genes[current]

        children = (inferred_phylogeny[left_child(current)],
                    inferred_phylogeny[right_child(current)])

        for child in children:
            total_start_length += len(current_gene)
            child_gene = genes[child]
            edits, decisions = levenshtein_distance_with_tracking(current_gene, child_gene)
            total_edits += edits

            for key in total_decisions:
                total_decisions[key] += decisions[key]

            # This check would fail if the tree wasn't full; but in this case we know it is
            if right_child(child) < results_length:
                queue.append(child)

    print 'Overall, encountered {edits} edits from a total starting length of {total}, P = {prob:.3f}'.format(
        edits=int(total_edits), total=total_start_length, prob=float(total_edits)/total_start_length
    )
    for key in total_decisions:
        dec = total_decisions[key]
        print 'Found {count} {key}, which are {dec_prob:.3f} of decisions and {prob:.3f} of overall length'.format(
            count=dec, key=key, dec_prob=float(dec)/total_edits, prob=float(dec)/total_start_length
        )


def brute_force_maximum_parsimony_tree(gene_indices, genes, resemblance_func, default=None):
    """
    Since we have such a limited gene pool, a brute-force approach to a maximum parsimony
    tree is feasible. We enumerate over permutations of the different gene indices, and
    compute the cost they would incur, using the resemblance func. The cost is defined
    as the cost from each parents to its children, so from the root to the two children,
    and from each child to its two grandchildren.

    In reality, we don't need to check all of of the n! permutations. Each fixing
    of root, children, and two grand children per child, is actually counted eight times,
    two different ordering for the grandchildren x 2 sides x two orderings of the children.
    However, the asymptotic behavior behavior remains O(n!), which happens to be feasible
    in this case and unfeasible in almost any real-world application.

    :param gene_indices: The indices of the different genes
    :param genes: The gene strings themselves
    :param resemblance_func: The resemblance metric used to build the tree. As this was
        written with parsimony in mind, the minimum is taken.
    :param default: A parameter passed to the resemblance function.
    :return: The permutation of the tree with incurred the minimal cost.
    """
    table = generate_similarity_table(gene_indices, genes, resemblance_func, default=default)

    min_cost = float('Inf')
    min_perm = None

    for p in permutations(gene_indices):
        cost = sum([table[p[i]][p[left_child(i)]] + table[p[i]][p[right_child(i)]]
                    for i in xrange(len(gene_indices)/2)])
        if cost < min_cost:
            min_perm = p
            min_cost = cost

    print min_cost, min_perm

    return min_perm


def main():
    gene_indices, genes = zip(*RAW_GENE_DATA)

    print_table(generate_similarity_table(gene_indices, genes), table_format='latex')
    print_table(generate_similarity_table(gene_indices, genes, proportion=True,
                                          thresholds=((1.0, ''), (0.9, '**'), (0.8, '*'),)),
                table_format='latex')
    greedy_phylogeny_inference(gene_indices, genes)

    print_table(generate_similarity_table(gene_indices, genes,
                                          resemblance_func=levenshtein_distance, default=0),
                table_format='latex')
    print_table(generate_similarity_table(gene_indices, genes, resemblance_func=levenshtein_distance,
                                          default=0, proportion=True,
                                          thresholds=((0, ''), (0.15, '**'), (0.25, '*'),), threshold_larger=False),
                table_format='latex')
    greedy_levenshtein_phylogeny = greedy_phylogeny_inference(
        gene_indices, genes, resemblance_func=lambda x, y: -1 * levenshtein_distance(x, y), default=0)
    levenshtein_probability_inferences(greedy_levenshtein_phylogeny, genes)

    levenshtein_mp_tree = brute_force_maximum_parsimony_tree(gene_indices, genes, levenshtein_distance, 0)
    levenshtein_probability_inferences(levenshtein_mp_tree, genes)

    brute_force_maximum_parsimony_tree(gene_indices, genes, lambda x, y: -1 * bottom_up_longest_subsequence(x, y))


if __name__ == '__main__':
    main()
    # print timeit(main, number=1)
