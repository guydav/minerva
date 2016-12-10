#!/usr/bin/env python
# -*- coding: utf-8 -*-
class Lesson(object):
    def __init__(self, name):
        self.name = name
        self.depends = []

    def requires(self, node):
        self.depends.append(node)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


def dep_resolve(node, resolved, unresolved):
    print str(node), resolved, unresolved
    unresolved.append(node)
    for dep in node.depends:
        if dep not in resolved:
            if dep in unresolved:
                raise Exception('Circular reference detected :’-( ')
            dep_resolve(dep, resolved, unresolved)
    resolved.append(node)


def calculate_syllabus():
    # Our lessons in alphabetical order:
    big_oh = Lesson("Big O notation")
    binary_trees = Lesson("Binary Search Trees")
    complexity = Lesson("Complexity")
    dynamic_p = Lesson("Dynamic Programming")
    heaps = Lesson("Heaps")
    heapsort = Lesson("Heap Sort")
    quicksort = Lesson("Quicksort")
    recursion = Lesson("Recursion")
    red_black = Lesson("Red Black trees")

    # The dependencies our lessons have:
    binary_trees.requires(recursion)
    binary_trees.requires(complexity)
    complexity.requires(big_oh)
    dynamic_p.requires(recursion)
    dynamic_p.requires(complexity)
    heaps.requires(recursion)
    heaps.requires(complexity)
    heapsort.requires(complexity)
    heapsort.requires(recursion)
    heapsort.requires(heaps)
    quicksort.requires(complexity)
    quicksort.requires(recursion)
    red_black.requires(binary_trees)

    # Our syllabus requires that we cover all lessons.
    syllabus = [complexity, big_oh, recursion, heaps, heapsort, quicksort,
                red_black, binary_trees, dynamic_p]

    resolved = []
    for lesson in syllabus:
        if lesson in resolved:
            continue
        dep_resolve(lesson, resolved, [])

    for node in resolved:
        print(node.name)

if __name__ == '__main__':
    calculate_syllabus()