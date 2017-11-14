import numpy as np

from tabulate import tabulate
from cs152.heap import MinHeap  # OOP wrapper for Python's heapq I wrote at some point
from functools import reduce, total_ordering
from collections import defaultdict
from itertools import product, combinations
import operator
import timeit
from cs152 import memoize


NEGATION_SYMBOLS = ('~', '￢', '-')


@total_ordering
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

    def __le__(self, other):
        # If need to sort, default to ordering by names
        return self.name < other.name


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


def DPLL_Satisfiable(knowledge_base, use_degree_heuristic=True,
                     use_pure_symbol=True):
    """
    Implementation of the DPLL algorithm as defined by Russel and Norwig in figure 7.17
    :param knowledge_base: The knowledge base (or sentence) whose satisfiability we wish
        to test, as a list of sets, each representing a CNF clause
    :return: (satisfiable, model), the former a truth value about whether or not the KB
        is satisfiable, the latter a model that satisfies it if one exists
    """
    symbols = reduce(lambda x, y: x.union(y), knowledge_base, set())
    full_symbols_copy = symbols.copy()

    if use_degree_heuristic:
        sums_and_symbols = [(np.sum([s in clause for clause in knowledge_base]), s)
                            for s in symbols]
        sums_and_symbols.sort()
        symbols = [symbol for (symbol_sum, symbol) in sums_and_symbols]

    # Convert KB to from list of sets to list of dicts
    # each clause dict containing symbol: sign
    # this is in many ways the same, but simplifies some work later on
    knowledge_base = [{s: s.sign for s in clause} for clause in knowledge_base]
    satisfiable, model = DPLL(knowledge_base, symbols, {}, use_pure_symbol)

    if satisfiable:
        output_model = {key: OUTPUT_MAPPING[val] for key, val in model.items()}
        for free_symbol in full_symbols_copy.difference(model.keys()):
            output_model[free_symbol] = FREE_OUTPUT

        for key in output_model:
            key.sign = True

        model = output_model

    return satisfiable, model


def pure_symbol_heuristic(clauses, symbols):
    """
    Iterator-based solution inspired by
    https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    We iterate through symbols in reverse order, since they're sorted in ascending order
    :return: The most recurring pure symbol
    """
    for symbol in reversed(symbols):
        signs = [clause[symbol] for clause in clauses if symbol in clause]
        iterator = iter(signs)
        try:
            first = next(iterator)
        except StopIteration:
            return symbol

        if all(first == rest for rest in iterator):
            return symbol

    return None


def unit_clause_heuristic(clauses, symbols, model):
    clauses_copy = [clause.copy() for clause in clauses]

    for symbol in model:
        for clause_index in reversed(range(len(clauses_copy))):
            clause = clauses_copy[clause_index]
            if symbol not in clause:
                continue

            # Signs already match, this clause is true, ignore it
            if model[symbol] == clause[symbol]:
                del clauses_copy[clause_index]

            else:
                del clause[symbol]

    # At this point, if we only filter to length-1 clauses
    # they should each be a dict with a single key
    unit_clauses = [clause for clause in clauses_copy if len(clause) == 1]
    return unit_clauses


def DPLL(clauses, symbols, model, use_pure_symbol=True, use_unit_clause=True):
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

    if use_pure_symbol:
        pure_symbol = pure_symbol_heuristic(clauses, symbols)
        if pure_symbol:
            symbols.remove(pure_symbol)
            model[pure_symbol] = pure_symbol.sign
            return DPLL(clauses, symbols.copy(), model)

    if use_unit_clause:
        # find all current unit clauses using the heuristic
        unit_clauses = unit_clause_heuristic(clauses, symbols, model)
        if len(unit_clauses) > 0:
            # create an update for each symbol, which is a set - if it's length 1,
            # we're fine, and we update, if it's longer, we tried to update multuple
            # values, and so we'll fail
            unit_clauses_update = defaultdict(set)
            [unit_clauses_update[symbol].add(clause[symbol])
             for clause in unit_clauses for symbol in clause]

            for symbol, value_set in unit_clauses_update.items():
                # conflict - we tried to assign both false and true to same literal
                if len(value_set) > 1:
                    return False, None

                symbols.remove(symbol)
                model[symbol] = value_set.pop()

            return DPLL(clauses, symbols.copy(), model)

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


