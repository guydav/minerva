#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tabulate
import numpy
from collections import defaultdict
from itertools import permutations


RAW_GENE_DATA = (
    (0, 'TCGCCAATATTATATTTTTGAAGGCGTAGCTAATGTGGATACTATGTAAGTCGCAAGCTCTGCCAAACAGGGCTAATGAACAAACACTATAATGAGGAC'),
    (1, 'TCGCAATATATCTTTTTGGAGAGCGTAGTATGTGGATATATCTCAAGTCGCAAGCTCTGCCGAAACAGGGCATGATTAAGGAATTACTACAATGAGGAAA'),
    (2, 'TGCAATATAACTTTTTAGGAGCATAGTAATGTCGGATATATTTTAAGTCGCAAGCGCCGAAACAGGGCACAATGAAGAAACACTATAAAGAGGAAC'),
    (3, 'TGCGCATATTTTTTTCTTAGTGTAGCGTAGTATGTGGAGATTCTCAAGTCGGAAGCTGCCTTAAACAGCGCATGATCAGGATTACTACAATAGAAGGAAA'),
    (4, 'TGCGCAATATATCTTTTTTAGAGAGCGTAGTATTGGTATATTCCCAAGTCGGAAGCTGCTTAAATCACCATGATACATGGAATTACTACAATAGAGGAAA'),
    (5, 'TCGCAATATATCTTTTTGAGAGCGTAGTAATGTGGATATATCTAAGTCGCAAGCTCTGCCGAAACAGGGCATAATGAAGAAACACTATAATGAGGAAC'),
    (6, 'TGCGCAATATATCTTTTTTAGGAGAGCGTAGTATGTGGATATTCTCAAGTCGGAAGCTGCCTTAAACAGCGCATGATAAGGAATTACTACAATAGAGGAAA'))
CACHE = defaultdict(dict)


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


@cache_decorator()
def bottom_up_longest_subsequence(x, y):
    m = len(x)
    n = len(y)

    cache = numpy.zeros((m + 1, n + 1))
    subsequences = numpy.chararray((m + 1, n + 1))
    subsequences[:] = ''

    for i in xrange(1, m + 1):
        for j in xrange(1, n + 1):
            up = cache[i - 1][j]
            left = cache[i][j - 1]

            if x[i - 1] == y[j - 1]:
                cache[i][j] = cache[i - 1][j - 1] + 1
                subsequences[i][j] = '\\'

            elif up >= left:
                cache[i][j] = up
                subsequences[i][j] = '^'
            else:
                cache[i][j] = left
                subsequences[i][j] = '<'

    return cache, subsequences


@cache_decorator(ignore_order=True)
def longest_subsequence_length(x, y):
    return bottom_up_longest_subsequence(x, y)[0][len(x)][len(y)]


def print_longest_subsequence(b, x, i, j):
    if 0 == i or 0 == j:
        return

    current = b[i][j]
    if '\\' == current:
        print_longest_subsequence(b, x, i - 1, j - 1)
        print x[i - 1],

    elif '^' == current:
        print_longest_subsequence(b, x, i - 1, j)

    # else:
    print_longest_subsequence(b, x, i, j - 1)


@cache_decorator(ignore_order=True)
def maximal_subsequence_length(x, y):
    return bottom_up_longest_subsequence(x, y)[0][len(x)][len(y)]


@cache_decorator()
def gene_resemblance(row_index, col_index, genes, resemblance_func, proportion=False, default=None):
    row_length = len(genes[row_index])
    result = row_length
    if default is not None:
        result = default

    if row_index != col_index:
        result = resemblance_func(genes[row_index], genes[col_index])

    if proportion:
        result = float(result) / row_length

    return result


def print_table(table, gene_indices=range(7)):
    header = [""]
    header.extend([str(i) for i in gene_indices])
    print tabulate.tabulate(table, header)


def add_threshold_marker(value, thresholds, larger=True):
    if thresholds is None:
        return value

    marker = ''

    for threshold_value, threshold_marker in thresholds:
        # TODO: think if there's an easy way to clean this up
        if larger:
            if value >= threshold_value:
                marker = threshold_marker
                break

        else:
            if value <= threshold_value:
                marker = threshold_marker
                break

    return '{value:.3f}{marker}'.format(value=value, marker=marker)


def generate_similarity_table(gene_indices, genes,
                              distance_func=longest_subsequence_length,
                              proportion=False, default=None, thresholds=None,
                              threshold_larger=True):
    table = [[row_gene] +
             [add_threshold_marker(gene_resemblance(
                 row_gene, col_gene, genes, distance_func, proportion, default), thresholds, threshold_larger)
              for col_gene in gene_indices]
             for row_gene in gene_indices]

    return table


def greedy_relationship_inference(gene_indices, genes, distance_func=longest_subsequence_length, default=None):
    table = [[gene_resemblance(row_gene, col_gene, genes, distance_func, default=default, proportion=True)
              for col_gene in gene_indices]
             for row_gene in gene_indices]

    # Although it's not mathematically correct, since all numbers are positive,
    # it doesn't harm to average in the 1.0 for the self-self proportion
    used = set()
    indices_set = set(gene_indices)

    averages = [(numpy.average(table[index]), index) for index in gene_indices]
    start = max(averages)[1]
    used.add(start)
    queue = [start]
    results = {}

    while queue and used != indices_set:
        current = queue.pop(0)
        children = [child[1] for child in
                    sorted([(table[current][i], i) for i in gene_indices if i not in used], reverse=True)[:2]]
        used.update(children)
        results[current] = children
        queue.extend(children)

    recursively_print_tree(results, start)
    return results, start


def recursively_print_tree(results, start, depth=0):
    # TODO: prettify this, or move to an actual tree, and print it even better
    print '\t' * depth + str(start)
    if start in results:
        for child in results[start]:
            recursively_print_tree(results, child, depth + 1)


DEL = 'deletions'
INS = 'insertions'
SUB = 'substitions'


@cache_decorator()
def levenshtein_distance_with_tracking(source, target):
    """
    Adapted from https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_full_matrix
    This is an implementation of the Wagner–Fischer algorithm, but tracking which "editing decision"
    is made at each point in time, in order to provide estimates thereof
    :param source: the source string
    :param target: the target string
    :return:
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


def levenshtein_distance(source, target):
    return levenshtein_distance_with_tracking(source, target)[0]


def levenshtein_probability_inferences(inference_results, start, gene_data=RAW_GENE_DATA):
    gene_indices, genes = zip(*gene_data)

    queue = [start]
    total_start_length = 0
    total_edits = 0
    total_decisions = {DEL: 0, INS: 0, SUB: 0}

    while queue:
        current = queue.pop(0)
        current_gene = genes[current]

        children = inference_results[current]

        for child in children:
            total_start_length += len(current_gene)
            child_gene = genes[child]
            edits, decisions = levenshtein_distance_with_tracking(current_gene, child_gene)
            total_edits += edits

            for key in total_decisions:
                total_decisions[key] += decisions[key]

            if child in inference_results:
                queue.append(child)

    print 'Overall, encountered {edits} edits from a total starting length of {total}, P = {prob:.3f}'.format(
        edits=total_edits, total=total_start_length, prob=float(total_edits)/total_start_length
    )
    for key in total_decisions:
        dec = total_decisions[key]
        print 'Found {count} {key}, which are {dec_prob:.3f} of decisions and {prob:.3f} of overall length'.format(
            count=dec, key=key, dec_prob=float(dec)/total_edits, prob=float(dec)/total_start_length
        )


def brute_force_maximum_parsimony_tree(distance_func, default=None, genes=RAW_GENE_DATA):
    gene_indices, genes = zip(*genes)
    table = [[gene_resemblance(row_gene, col_gene, genes, distance_func, default=default)
              for col_gene in gene_indices]
             for row_gene in gene_indices]

    min_cost = float('Inf')
    min_perm = None

    for p in permutations(gene_indices):
        cost = sum([table[p[i]][p[2 * i + 1]] + table[p[i]][p[2 * i + 2]] for i in xrange(3)])
        if cost < min_cost:
            min_perm = p
            min_cost = cost

    print min_cost, min_perm

    min_perm_as_dict = {min_perm[i]: (min_perm[2 * i + 1], min_perm[2 * i + 2]) for i in xrange(3)}
    return min_perm_as_dict, min_perm[0]


def main():
    gene_indices, genes = zip(*RAW_GENE_DATA)

    print_table(generate_similarity_table(gene_indices, genes))
    print_table(generate_similarity_table(gene_indices, genes, proportion=True,
                                          thresholds=((1.0, ''), (0.9, '**'), (0.8, '*'),)))
    results, start = greedy_relationship_inference(gene_indices, genes)

    print_table(generate_similarity_table(gene_indices, genes, distance_func=levenshtein_distance, default=0))
    print_table(generate_similarity_table(gene_indices, genes, distance_func=levenshtein_distance,
                                          default=0, proportion=True,
                                          thresholds=((0, ''), (0.15, '**'), (0.25, '*'),), threshold_larger=False))
    results, start = greedy_relationship_inference(gene_indices, genes,
                                                   distance_func=lambda x, y: -1 * levenshtein_distance(x, y),
                                                   default=0)
    levenshtein_probability_inferences(results, start)
    levenshtein_mp_tree, levenshtein_mp_root = \
        brute_force_maximum_parsimony_tree(levenshtein_distance, 0)
    levenshtein_probability_inferences(levenshtein_mp_tree, levenshtein_mp_root)

    lcs_mp_tree, lcs_root = \
        brute_force_maximum_parsimony_tree(lambda x, y: -1 * longest_subsequence_length(x, y))


if __name__ == '__main__':
    main()
    # print timeit(main, number=1)
