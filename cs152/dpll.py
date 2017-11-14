import numpy as np

from tabulate import tabulate
from cs152.heap import MinHeap  # OOP wrapper for Python's heapq I wrote at some point
from functools import reduce
from itertools import product, combinations
import operator
import timeit
from cs152 import memoize


NEGATION_SYMBOLS = ('~', '￢', '-')


class Literal:
    def __init__(self, name, sign=True):
        if name[0] in NEGATION_SYMBOLS:
            self.name = name[1:]
            self.sign = False

        else:
            self.name = name
            self.sign = sign

    def __neg__(self):
        self.sign = not self.sign

    def __repr__(self):
        return '{sign}{name}'.format(sign='' if self.sign else '￢', name=self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not type(other) == Literal:
            return False

        return self.name == other.name


class UndecidableException(Exception):
    """
    A custom exception to handle the case of an evaluation being currently undeciable -
     for example, because one of the variables was not offered a value
    """
    pass


def _evaluate(clause, model):
    """
    Helper method to evaluate a CNF clause in a model.
    The clause and the model should both be sets of Literals.
    :param clause: The clause (disjunction of literals) as a set of Literals
    :param model: The model as a dictionary from literal to true value
    :return: True if the sentence evaluates to true in the model, at least one of the literals
        is true. False if the sentence evaluates to false in the model. Raises
    """
    false_count = 0
    for literal in clause:
        if literal in model:
            # this is a disjunction, we need just a single true
            if literal.sign == model[literal]:
                return True

            false_count += 1

    if false_count == len(clause):
        return False

    raise UndecidableException()


TRUE_OUTPUT = 'true'
FALSE_OUTPUT = 'false'
FREE_OUTPUT = 'free'
OUTPUT_MAPPING = {True: TRUE_OUTPUT, False: FALSE_OUTPUT}


def DPLL_Satisfiable(knowledge_base):
    """
    Implementation of the DPLL algorithm as defined by Russel and Norwig in figure 7.17
    :param knowledge_base: The knowledge base (or sentence) whose satisfiability we wish
        to test, as a list of sets, each representing a CNF clause
    :return: (satisfiable, model), the former a truth value about whether or not the KB
        is satisfiable, the latter a model that satisfies it if one exists
    """
    symbols = reduce(lambda x, y: x.union(y), knowledge_base, set())
    full_symbols_copy = symbols.copy()
    satisfiable, model = DPLL(knowledge_base, symbols, {})
    if satisfiable:
        output_model = {key: OUTPUT_MAPPING[val] for key, val in model.items()}
        for free_symbol in full_symbols_copy.difference(model.keys()):
            output_model[free_symbol] = FREE_OUTPUT

        for key in output_model:
            key.sign = True

        model = output_model

    return satisfiable, model


def DPLL(clauses, symbols, model):
    true_count = 0
    for clause in clauses:
        try:
            if not _evaluate(clause, model):
                return False, None

            true_count += 1

        except UndecidableException:
            continue

    if len(clauses) == true_count:
        return True, model

    current = symbols.pop()
    model_false = model.copy()
    model[current] = True
    model_false[current] = False

    # The or of tuples behaves in unfortunate ways, sadly
    true_model_tv, true_model = DPLL(clauses, symbols.copy(), model)
    if true_model_tv:
        return true_model_tv, true_model

    false_model_tv, false_model = DPLL(clauses, symbols.copy(), model_false)
    if false_model_tv:
        return false_model_tv, false_model

    return False, None


if __name__ == '__main__':
    test_KB = [{Literal('A'), Literal('B')},
               {Literal('A'), Literal('C', False)},
               {Literal('A', False), Literal('B'), Literal('D')},
               ]

    # print(DPLL_Satisfiable(test_KB))

    # TODO: write nicely in Markdown
    """
    ** A <=> (B v E)
    [A => (B v E)] ^ [(B v E) => A]
    [~A v B v E] ^ [~(B v E) v A]
    [~A v B v E] ^ [(~B ^ ~E) v A]
    [~A v B v E] ^ (~B v A) ^ (~E v A)
    
    ** E => D
    ~E v D
    
    ** C ^ F => ~B
    ~(C ^ F) v ~B
    ~C v ~F v ~B
    
    ** E => B
    ~E v B
    
    ** B => F
    ~B v F
    
    ** B => C
    ~B v C
    
    A: false
    B: false
    C:
    D:
    E: false
    F:
    """

    RN_7_20_KB = [
        {Literal('~A'), Literal('B'), Literal('E')},
        {Literal('~B'), Literal('A')},
        {Literal('~E'), Literal('A')},
        {Literal('~E'), Literal('D')},
        {Literal('~C'), Literal('~F'), Literal('~B')},
        {Literal('~E'), Literal('B')},
        {Literal('~B'), Literal('F')},
        {Literal('~B'), Literal('C')}
    ]

    # test_model = {Literal('A'): False,
    #          Literal('B'): False,
    #          Literal('E'): False}

    # for c in RN_7_20_KB:
    #     try:
    #         print(c, _evaluate(c, test_model))
    #     except UndecidableException:
    #         print(c, 'undecidable')

    print(DPLL_Satisfiable(RN_7_20_KB))


