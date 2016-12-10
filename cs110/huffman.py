#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq
import urllib
import os
from collections import defaultdict
from bitarray import bitarray
import cPickle as pickle
from itertools import groupby
import numpy


CHILD_ON_LEFT = True
CHILD_ON_RIGHT = False


class Node(object):
    def __init__(self, data, l_child=None, r_child=None):
        self.data = data
        self.l_child = l_child
        self.r_child = r_child

    def __repr__(self):
        return '<{data}>'.format(data=self.data)


# Download the file if need be:
def download_file(url, filename):
    if not os.path.exists(filename):
        urllib.urlretrieve(url + filename, filename)


# build a frequency table:
def build_freq(filename):
    freq = defaultdict(int)
    with open(filename, 'r') as f:
        for line in f:
            for char in line.decode('utf-8-sig'):
                freq[char] += 1
    total = float(sum(freq.values()))
    return {char: count / total for (char, count) in freq.items()}


# Now build the Huffman encoding:
def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights.

    Accept a dictionary which maps a symbol to a probability.
    Return a new dictionary which maps a symbol to a bitarray."""

    symbol_heap = []
    for symbol in symb2freq:
        heapq.heappush(symbol_heap, (symb2freq[symbol], Node(symbol)))

    while len(symbol_heap) > 1:
        left_count, left_node = heapq.heappop(symbol_heap)
        right_count, right_node = heapq.heappop(symbol_heap)
        new_node = Node(left_count + right_count, left_node, right_node)
        heapq.heappush(symbol_heap, (new_node.data, new_node))

    _, root = heapq.heappop(symbol_heap)
    print_tree(root)
    return _create_symbol_dict(root)


def print_tree(root, depth=0, is_right=None):
    if root is None:
        return

    if depth > 20:
        raise ValueError("Unable to comply, building in progress")

    if root.r_child:
        print_tree(root.r_child, depth + 1, CHILD_ON_RIGHT)

    if is_right is None:
        symbol = '--'
    elif is_right == CHILD_ON_RIGHT:
        symbol = '/'
    else:
        symbol = '\\'

    print '   ' * depth + ' {symbol} {data}'.format(symbol=symbol, data=root.data)

    if root.l_child:
        print_tree(root.l_child, depth + 1, CHILD_ON_LEFT)


def _create_symbol_dict(node, prefix=None, symbol_dict=None):
    if symbol_dict is None:
        symbol_dict = {}

    if prefix is None:
        prefix = bitarray()

    if node.l_child is None and node.r_child is None:
        symbol_dict[node.data] = prefix

    else:
        if node.l_child is not None:
            _create_symbol_dict(node.l_child, prefix + bitarray('0'), symbol_dict)

        if node.r_child is not None:
            _create_symbol_dict(node.r_child, prefix + bitarray('1'), symbol_dict)

    return symbol_dict


# Now compress the file:
def compress(filename, encoding, compressed_name=None):
    if compressed_name is None:
        compressed_name = filename + ".huff"
    output = bitarray()
    with open(filename, 'r') as f:
        for line in f:
            for char in line.decode('utf-8-sig'):
                output.extend(encoding[char])
    N = len(output)
    with open(compressed_name, 'wb') as f:
        pickle.dump(N, f)
        pickle.dump(encoding, f)
        output.tofile(f)


# Now decompress the file:
def decompress(filename, decompressed_name=None):
    if decompressed_name is None:
        decompressed_name = filename + ".dehuff"
    with open(filename, 'rb') as f:
        N = pickle.load(f)
        encoding = pickle.load(f)
        bits = bitarray()
        bits.fromfile(f)
        bits = bits[:N]

    # Totally cheating here and using a builtin method:
    output = bits.decode(encoding)

    output = "".join(output).encode('utf-8-sig')
    with open(decompressed_name, 'wb') as f:
        f.write(output)


def main():
    url = "https://www.gutenberg.org/ebooks/"
    filename = "100.txt.utf-8"

    download_file(url, filename)
    freq = build_freq(filename)
    encoding = encode(freq)
    compress(filename, encoding)
    decompress(filename + ".huff")


def run_length_coding(string):
    return ''.join(['{}{}'.format(k, sum(1 for _ in g)) for k, g in groupby(string)])


def entropy(symbols):
    return sum([-1 * p * numpy.log2(p) for p in symbols.values()])


if __name__ == '__main__':
    symbols = {'a': 0.25, 'b': 0.5, 'c': 0.125, 'd': 0.125}
    print encode(symbols)
    print entropy(symbols)
    print entropy({'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25})
    # bent_coin = numpy.random.binomial(1, 0.99, 10000)
    # print run_length_coding(''.join([x == 1 and 'H' or 'T' for x in bent_coin]))
    # main()
