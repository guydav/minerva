import numpy as np
from functools import reduce, total_ordering
from collections import defaultdict
from itertools import product
import timeit
import tabulate


timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


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

    def copy(self):
        return Literal(self.name, self.sign)


class UndecidableException(Exception):
    """
    A custom exception to handle the case of an evaluation being currently undeciable -
     because one of the variables was not offered a value
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
                     use_pure_symbol=True, use_unit_clause=True):
    """
    Implementation of the DPLL algorithm as defined by Russel and Norwig in figure 7.17
    :param knowledge_base: The knowledge base (or sentence) whose satisfiability we wish
        to test, as a list of sets, each representing a CNF clause
    :return: (satisfiable, model), the former a truth value about whether or not the KB
        is satisfiable, the latter a model that satisfies it if one exists
    """
    symbols = reduce(lambda x, y: x.union(y), knowledge_base, set())
    full_symbols_copy = symbols.copy()

    if use_degree_heuristic or use_pure_symbol:
        sums_and_symbols = [(np.sum([s in clause for clause in knowledge_base]), s)
                            for s in symbols]
        sums_and_symbols.sort()
        symbols = [symbol for (symbol_sum, symbol) in sums_and_symbols]

    # Convert KB to from list of sets to list of dicts
    # each clause dict containing symbol: sign
    # this is in many ways the same, but simplifies some work later on
    knowledge_base = [{s: s.sign for s in clause} for clause in knowledge_base]
    satisfiable, model = DPLL(knowledge_base, symbols, {},
                              use_pure_symbol, use_unit_clause)

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
        if len(signs) == 0:
            continue

        iterator = iter(signs)
        try:
            first = next(iterator)
        except StopIteration:
            return symbol

        if all(first == rest for rest in iterator):
            return symbol

    return None


def unit_clause_heuristic(clauses, model):
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
            return DPLL(clauses, symbols.copy(), model,
                        use_pure_symbol, use_unit_clause)

    if use_unit_clause:
        # find all current unit clauses using the heuristic
        unit_clauses = unit_clause_heuristic(clauses, model)
        if len(unit_clauses) > 0:
            # create an update for each symbol, which is a set - if it's length 1,
            # we're fine, and we update, if it's longer, we tried to update
            # a literal to both true and false, and so we'll fail
            unit_clauses_update = defaultdict(set)
            [unit_clauses_update[symbol].add(clause[symbol])
             for clause in unit_clauses for symbol in clause]

            for symbol, value_set in unit_clauses_update.items():
                # conflict - we tried to assign both false and true to same literal
                if len(value_set) > 1:
                    return False, None

                symbols.remove(symbol)
                model[symbol] = value_set.pop()

            return DPLL(clauses, symbols.copy(), model,
                        use_pure_symbol, use_unit_clause)

    # If we arrived at this point, pick a symbol at random
    current = symbols.pop()
    model_false = model.copy()
    model[current] = True
    model_false[current] = False

    # The or of tuples behaves in unfortunate ways, sadly
    true_model_tv, true_model = DPLL(clauses, symbols.copy(), model,
                                     use_pure_symbol, use_unit_clause)
    false_model_tv, false_model = DPLL(clauses, symbols.copy(), model_false,
                                       use_pure_symbol, use_unit_clause)

    if true_model or false_model_tv:
        return True, true_model if true_model_tv else false_model

    return False, None


HEADERS = ('Degree H', 'Pure Symbol H', 'Unit Clause H', 'Time', 'Result', 'Avg. Free Symbol Count')


def run_with_heuristic_combinations(knowledge_base, times=10):
    results = []
    for use_degree, use_pure, use_unit in product([False, True], repeat=3):
        def dpll():
            return DPLL_Satisfiable(knowledge_base, use_degree, use_pure, use_unit)

        # intentionally not using timeit's number argument, since I want to average
        # the number of free symbols left
        total_time, total_free_symbols = 0, 0
        for _ in range(times):
            infer_time, (result, model) = timeit.timeit(dpll, number=times)
            total_time += infer_time
            if model:
                total_free_symbols += sum(map(lambda v: v == FREE_OUTPUT, model.values()))

        results.append((use_degree, use_pure, use_unit, total_time, result, total_free_symbols / times))

    print(tabulate.tabulate(results, HEADERS, tablefmt='fancy_grid'))


DEFAULT_KB_SIZE = 40
DEFAULT_NUM_LITERALS = 20
DEFAULT_LITERALS_IN_CLAUSE_MEAN = 4


def generate_satisfiable_knowledge_base(should_flip=True, kb_size=DEFAULT_KB_SIZE,
                                        num_literals=DEFAULT_NUM_LITERALS,
                                        clause_mean=DEFAULT_LITERALS_IN_CLAUSE_MEAN):
    """
    It's actually fairly easy to generate a satisfiable knowledge base:
    1) Create literals with random signs
    2) Sample a random combination of them to be in a clause together.
    These KBs are fairly naive, but they're guaranteed to be satisfiable.
    And in fact, so long as at least one literal is kept in its original state,
    we can flip some literals in each clause and remain satisfiable
    """
    literals = [Literal(str(i), np.random.random() > 0.5) for i in range(num_literals)]
    knowledge_base = []
    for _ in range(kb_size):
        clause_size = max(np.random.binomial(num_literals, clause_mean / num_literals), 1)
        clause = [l.copy() for l in np.random.choice(literals, clause_size, replace=False)]
        if should_flip:
            flip_count = np.random.randint(0, clause_size)  # half-open interval
            [-l for l in np.random.choice(literals, flip_count, replace=False)]

        knowledge_base.append(set(clause))

    return knowledge_base


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

    run_with_heuristic_combinations(RN_7_20_KB)

    run_with_heuristic_combinations(generate_satisfiable_knowledge_base(False), times=1)

    run_with_heuristic_combinations(generate_satisfiable_knowledge_base(True), times=1)

