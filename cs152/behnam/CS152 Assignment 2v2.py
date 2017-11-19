def neg(symbol):

    if symbol[0] == "-":
        return symbol[1:]
    else:
        return "-" + symbol

def pure(symbol):
    if symbol[0] == "-":
        return symbol[1:]
    else:
        return symbol

def output(s1,s2):
    dictionary = {}
    for i in s1:
        if neg(i) not in dictionary:
            dictionary[pure(i)] = str(not((i) == "-" + i[1:]))
    for i in s2:
        if pure(i) not in dictionary:
            dictionary[pure(i)] = "Free"
    return dictionary

def order_and_unit(KB):
    symbols = dict()
    unit_clauses = []
    for i in KB:
        for j in i:
            if ((j in symbols) == False):
                symbols[j] = 1

                #this prioritizes unit clauses: essentially, if a symbol is
                # in a unit clause, we know the clause is length 1. Adding len(KB)
                # to the count ensures that it gets priority over any symbol that
                # isn't a unit clause, even if that symbol appears in every other
                # clause. If there are multiple unit clauses, however, we can also
                # rest assured that frequency will dictate which of the unit clause
                # symbols is prioritized first.
                if (len(i) == 1) and (symbols[j]<len(KB)):
                    symbols[j] = symbols[j] + len(KB)
                    unit_clauses.append(j)

            # This elif is tacky but accounts for the case where we find a unit
            # clause with a symbol we've already put in our dictionary. It is
            # otherwise identical to the same conditional above.
            elif (len(i) == 1) and (symbols[j]<len(KB)):
                symbols[j] = symbols[j] + len(KB)
                unit_clauses.append(j)
            else:
                symbols[j] = symbols[j] + 1
    symbol_list = []

    # Now we sort our dictionary by value and append the keys to a list.
    for key, value in sorted(symbols.items(), key=lambda item: (item[1], item[0])):
        if (neg(key) in symbol_list) == False:
            symbol_list.append(key)
    symbol_list.reverse()
    for c in unit_clauses:
        if (neg(c) in symbol_list) == True:
            symbol_list.remove(neg(c))
    return symbol_list

def DPLL_Satisfiable(KB):

    return DPLL(KB,order_and_unit(KB),[],True)

def DPLL(clauses,symbols,model,contains_unit_clause):

    #Base case of DPLL
    true_count = 0
    true_clauses = []
    for clause in clauses:

        #I use false_count to keep track of how many symbols in my model
        # are false
        false_count = 0
        for symbol in model:
            if (symbol in clause) == True:
                true_count += 1
                true_clauses.append(clause)
                break

            #this checks for if the negation of the symbol is present in the clause.
            # If it is, we get suspicious, but note that every symbol in the clause
            # must evaluate as false (i.e. be the opposite symbol) for the clause itself to evaluate as false.
            elif (("-"+symbol in clause) == True) or ((symbol[1:] in clause) == True):
                false_count += 1
        if false_count == len(clause):
            return False


    if true_count == len(clauses):

        return True, output(model,symbols)
    for i in true_clauses:
        clauses.remove(i)

    if len(symbols) != 0:

        nextsymbol = symbols.pop(symbols.index(symbols[0]))
        #if
        mod1 = model + [nextsymbol]
        mod2 = model +[neg(nextsymbol)]

        return DPLL(clauses,symbols,mod2,contains_unit_clause) or DPLL(clauses,symbols,mod1,contains_unit_clause)
    else:
        return False


kb = [["-A","-B"],
    ["-A","-C"],
    ["A","B","D"]]

kb2 = [["-A", "B", "E"],
    ["A", "-B"],
    ["A", "-E"],
    ["-E", "D"],
    ["-C", "-F", "-B"],
    ["-E", "B"],
    ["-B", "F"],
    ["-B", "C"] ]
print(DPLL_Satisfiable(kb2))

