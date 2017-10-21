import numpy as np
from collections import defaultdict
from functools import total_ordering


@total_ordering
class Symbol:
    def __init__(self, symbol, value=None):
        self.symbol = symbol
        self.negated = value

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __lt__(self, other):
        return self.symbol < other.symbol

    def __repr__(self):
        return str(self.symbol)


class DefiniteClause:
    def __init__(self, conclusion, premise=None):
        if type(conclusion) is not Symbol:
            conclusion = Symbol(conclusion)
        self.conclusion = conclusion

        if premise is None:
            self.premise = set()
        else:
            self.premise = set([p if type(p) == Symbol else Symbol(p) for p in premise])

    def __hash__(self):
        return hash(tuple(list(sorted(self.premise)) + [self.conclusion]))

    def __repr__(self):
        return ' ^ '.join([str(p) for p in self.premise]) + ' => ' + str(self.conclusion)


def forward_chaining_entails(knowledge_base, query=None):
    count = {clause: len(clause.premise) for clause in knowledge_base}
    inferred = defaultdict(lambda: False)

    symbols_to_clauses = defaultdict(set)
    for clause in knowledge_base:
        for symbol in clause.premise:
            symbols_to_clauses[symbol].add(clause)

    agenda = []
    for clause in knowledge_base:
        if 0 == len(clause.premise):
            agenda.append(clause.conclusion)
            inferred[clause.conclusion] = True

    while len(agenda):
        current = agenda.pop()
        # slightly changed form the pseudocode in Russell & Norwig
        for clause in symbols_to_clauses[current]:
            count[clause] -= 1
            if 0 == count[clause] and not inferred[clause.conclusion]:
                inferred[clause.conclusion] = True
                if query is not None and clause.conclusion == query:
                    return True

                agenda.append(clause.conclusion)

    return inferred

if __name__ == '__main__':
    knowledge_base = [
        DefiniteClause("A", ["B", "C"]),
        DefiniteClause("B", ["D"]),
        DefiniteClause("B", ["E"]),
        DefiniteClause("D", ["H"]),
        DefiniteClause("F", ["G", "B"]),
        DefiniteClause("G", ["C", "K"]),
        DefiniteClause("J", ["A", "B"]),
        DefiniteClause("J", ["A", "B"]),
        DefiniteClause("C"),
        DefiniteClause("E"),
    ]

    inferred = forward_chaining_entails(knowledge_base)
    print(sorted([symbol for symbol in inferred if inferred[symbol]]))

