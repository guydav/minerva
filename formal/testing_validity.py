__author__ = 'guydavidson'

# Cnstants:
SINGLE_COMPONENT_COUNT = 1
DEFAULT_COMPONENT_COUNT = 2
CONJUNCTION_REPR_OPERATOR = ' & '
DISJUNCTION_REPR_OPERATOR = ' v '

class Connective(object):
    '''
    Assumes an expression of some parts, with a specific manner of evaluation.

    Subclasses must implement _evaluate, and should also implement __repr__
    '''
    def __init__(self, count, *components):
        if not (len(components) == count):
            # CodeSkulptor doesn't support string.format
            # raise ValueError(
            #     '{class_name} created with an invalid number of arguments - expected {count}, found:\n{components}'.format(
            #     class_name=self.__class__, count=count,components=components))
            raise ValueError('%s created with an invalid number of arguments - expected %s, found:\n%s' %
                             (self.__class__, count, components))


        self.components = components

    def __call__(self, values):
        '''
        _evaluate the logical expression - _evaluate should only be called internally
        :param values: a dictionary mapping sentence letters to truth-values
        :return: the truth-value of the entire argument
        '''
        self.evaluated_components = []
        for component in self.components:
            if isinstance(component, Connective):
                self.evaluated_components.append(component(values))

            # CodeSkulptor doesn't support basestring
            # elif isinstance(component, basestring) and component in values:
            elif isinstance(component, str) and component in values:
                self.evaluated_components.append(values[component])

            else:
                # CodeSkulptor doesn't support string.format
                # raise ValueError('Component {component} is not a connective or did not exists in values:\n{values}'.format(
                #     component=component, values=values))
                raise ValueError('Component %s is not a connective or did not exists in values:\n%s' %
                                 (component, values))

        return self._evaluate()

    def _evaluate(self):
        '''
        _evaluate assumes it's called after self.evaluated_components is filled with boolean values
        for the different components
        :return: The result of evaluating the expression on the different values
        '''
        raise NotImplemented('You must implement this method when subclassing')

class Variable(Connective):
    '''
    This class describes a premise which is just a single variable
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        Connective.__init__(self, SINGLE_COMPONENT_COUNT, *components)

    def _evaluate(self):
        return self.evaluated_components[0]

    def __repr__(self):
        # CodeSkulptor doesn't support string.format
        # return '{component}'.format(component=self.components[0])
        return '%s' % (self.components[0],)

class Negation(Connective):
    '''
    This class describes a negation of whichever component is receives
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        Connective.__init__(self, SINGLE_COMPONENT_COUNT, *components)

    def _evaluate(self):
        return not self.evaluated_components[0]

    def __repr__(self):
        # CodeSkulptor doesn't support string.format
        # return '~({component})'.format(component=self.components[0])
        return '~(%s)' % (self.components[0],)

# Shorter name
Not = Negation

class VariableArgumentConnective(Connective):
    '''
    Assumes an expression of some parts, with a specific manner of evaluation.

    Subclasses must implement _evaluate, and should also implement __repr__
    '''
    def __init__(self, min_count, repr_operator, reduce_lambda, *components):
        # Intentionally not calling super in this case
        if not (len(components) >= min_count):
            # CodeSkulptor doesn't support string.format
            # raise ValueError(
            #     '{class_name} created with an invalid number of arguments - expected {count}, found:\n{components}'.format(
            #     class_name=self.__class__, count=count,components=components))
            raise ValueError('%s created with an invalid number of arguments - expected %s, found:\n%s' %
                             (self.__class__, min_count, components))

        self.components = components
        self.repr_operator = repr_operator
        self.reduce_lambda = reduce_lambda

    def _evaluate(self):
        return reduce(self.reduce_lambda, self.evaluated_components)

    def __repr__(self):
        # CodeSkulptor doesn't support string.format
        # return '({first} & {second})'.format(first=self.components[0], second=self.components[1])
        # return '(%s & %s)' % (self.components[0], self.components[1])
        return '(%s)' % (self.repr_operator.join([str(c) for c in self.components]))

class Conjunction(VariableArgumentConnective):
    '''
    This class describes a boolean and (conjunction) between two components.
    Returns True when both components evaluate to True, and False otherwise
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        VariableArgumentConnective.__init__(self, DEFAULT_COMPONENT_COUNT, CONJUNCTION_REPR_OPERATOR,
                                            lambda a, b: (a and b), *components)

    # def _evaluate(self):
    #     return reduce(lambda a, b: a and b, self.components)

    # def __repr__(self):
    #     # CodeSkulptor doesn't support string.format
    #     # return '({first} & {second})'.format(first=self.components[0], second=self.components[1])
    #     # return '(%s & %s)' % (self.components[0], self.components[1])
    #     return '(%s)' % (self.repr_operator.join(self.components))

# Shorter name
And = Conjunction

class Disjunction(VariableArgumentConnective):
    '''
    This class describes a boolean or (disjunction) between two components
    Returns False when both components evaluate to False, and True otherwise.
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        VariableArgumentConnective.__init__(self, DEFAULT_COMPONENT_COUNT, DISJUNCTION_REPR_OPERATOR,
                                            lambda a, b: (a or b), *components)

    # def _evaluate(self):
    #     return self.evaluated_components[0] or self.evaluated_components[1]
    #
    # def __repr__(self):
    #     # CodeSkulptor doesn't support string.format
    #     # return '({first} v {second})'.format(first=self.components[0], second=self.components[1])
    #     return '(%s v %s)' % (self.components[0], self.components[1])

# Shorter name
Or = Disjunction

class ExclusiveDisjunction(Connective):
    '''
    This class describes a boolean xor (exclusive disjunction) between two components
    Returns False when both components evaluate to be equal to each other, and True otherwise.
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        Connective.__init__(self, DEFAULT_COMPONENT_COUNT, *components)

    def _evaluate(self):
        return not (self.evaluated_components[0] == self.evaluated_components[1])

    def __repr__(self):
        # CodeSkulptor doesn't support string.format
        # return '({first} ^ {second})'.format(first=self.components[0], second=self.components[1])
        return '(%s ^ %s)' % (self.components[0], self.components[1])

# Shorter name
Xor = ExclusiveDisjunction

class Conditional(Connective):
    '''
    This class describes a boolean conditional (if ... then ...) between two components
    Returns False when the first component evaluates to True and the second to False, and True otherwise.
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        Connective.__init__(self, DEFAULT_COMPONENT_COUNT, *components)

    def _evaluate(self):
        return (not self.evaluated_components[0]) or self.evaluated_components[1]

    def __repr__(self):
        # CodeSkulptor doesn't support string.format
        # return '({first} -> {second})'.format(first=self.components[0], second=self.components[1])
        return '(%s -> %s)' % (self.components[0], self.components[1])

class Biconditional(Connective):
    '''
    This class describes a boolean biconditional (... if and only if ...) between two components
    Returns True when both components evaluate to the same value, and False otherwise
    '''
    def __init__(self, *components):
        # super() not implemented in CodeSkulptor - hardcoding the super class
        Connective.__init__(self, DEFAULT_COMPONENT_COUNT, *components)

    def _evaluate(self):
        return self.evaluated_components[0] == self.evaluated_components[1]

    def __repr__(self):
        # CodeSkulptor doesn't support string.format
        # return '({first} <--> {second})'.format(first=self.components[0], second=self.components[1])
        return '(%s <--> %s)' % (self.components[0], self.components[1])

def _generate_permutations(variables):
    '''
    Generates all true/false permutations of the given variable list. Each dictionary in the returned list represents
    one line in the truth table.
    :param variables: A list of boolean variable names, as strings
    :return: a list of dictionaries, each representing a permutation of the possible values of the variables.
    '''
    permutations = []
    for variable in variables:
        if not permutations:
            permutations.append({variable: True})
            permutations.append({variable: False})

        else:
            for permutation in permutations:
                if not variable in permutation:
                    # dict.copy not implement in CodeSkulptor
                    # permutation_copy = permutation.copy()
                    permutation_copy = dict(permutation)
                    permutation[variable] = True
                    permutation_copy[variable] = False
                    # Don't need to append permutation, it's already there
                    permutations.append(permutation_copy)

    # This step shouldn't be necessary, here just as a safety - really isn't necessary
    # permutations = [p for p in permutations if len(p) == len(variables)]
    return permutations

def valid(premises, conclusion, variables):
    '''
    Tests the validity of a logical argument. Fails only if all premises could be true and the conclusion false.
    See main() for usage examples.
    :param premises: A list of classes which inherit from Connective, potentially combined together.
    :param conclusion: An object which inherits from Connective.
    :param variables: A list of the boolean variables used in the premises and conclusion
    :return: False if the argument is invalid, True if it is, with an informative printed message.
    '''
    # Generate all permutation of the variables, for a truth table
    permutations = _generate_permutations(variables)

    # Check each row in the truth table
    for permutation in permutations:
        # Evaluate each premise for the given permutation
        premise_values = [premise(permutation) for premise in premises]

        # Test if all premises are true
        if reduce(lambda a,b: a and b, premise_values):
            conclusion_value = conclusion(permutation)

            if not conclusion_value:
                # CodeSkulptor doesn't support string.format
                # print 'For the values: {values}, the premises {premises} are all true, ' \
                #       'while the conclusion {conclusion} is false, and hence the argument is invalid'.format(
                #     values=permutation, premises=premises, conclusion=conclusion)
                print 'For the values: %s, the premises %s are all true, while the conclusion %s is false, ' \
                      'and hence the argument is invalid' % (permutation, premises, conclusion)
                return False

    # CodeSkulptor doesn't support string.format
    # print 'After evaluating the entire truth table, the premises {premises} ' \
    #       'do not contradict the conclusion {conclusion}, and hence the argument is valid'.format(
    #     premises=premises, conclusion=conclusion)
    print 'After evaluating the entire truth table, the premises %s do not contradict the conclusion %s, ' \
          'and hence the argument is valid' % (premises, conclusion)
    return True

def test_with_message(message, premises, conclusion, variables):
    '''
    A wrapper function fod valid(premises, conclusion, variables).
    Prints a message, runs a validity test, prints an empty line
    :param message: The message to be printed before the test runs
    :param premises: The premises to be tested (same as valid())
    :param conclusion: The conclusion to be tested (same as valid())
    :param variables: The variables used in the premises and conclusion (same as valid())
    :return: None
    '''
    print message
    valid(premises, conclusion, variables)
    print

def main():
    '''
    A main function which demonstrates how to best use the Connective classes and valid function to test the validity
    of argument.
    :return: None
    '''
    test_with_message('Testing modus ponens:',
        [Conditional('P', 'Q'),
         Variable('P')],
        Variable('Q'),
        ['P', 'Q'])

    test_with_message('Testing modus tollens:',
        [Conditional('P', 'Q'),
         Negation('Q')],
        Negation('P'),
        ['P', 'Q'])

    test_with_message('Testing disjunctive syllogism:',
        [Disjunction('P', 'Q'),
         Negation('P')],
        Variable('Q'),
        ['P', 'Q'])

    test_with_message('Testing the complicated case:',
        [Conditional(Conjunction('A', 'B'), Disjunction(Negation('A'), 'B')),
         Conditional('B', Negation('A'))],
        Disjunction('A', Negation('B')),
        ['A', 'B'])

    test_with_message('Testing with three premises and variables:',
        [Variable('A'),
         Variable('B'),
         Variable('C')],
        Disjunction('A', Disjunction('B', 'C')),
        ['A', 'B', 'C'])

    test_with_message('Testing XOR and biconditional:',
        [ExclusiveDisjunction('A', 'B'),
         Biconditional('A', 'B')],
        Disjunction('A', Negation('B')),
        ['A', 'B'])

    test_with_message('Another, more complicated example:',
        [Conditional('A', Conjunction('B', Negation('C'))),
         Conditional('B', Conjunction('A', 'C'))],
        Conjunction(Negation('A'), Negation('B')),
        ['A', 'B', 'C'])

    test_with_message('And just for fun, an example from the quiz:',
        [Conditional(Disjunction('L', Negation('T'), 'J'), Variable('K')),
         Conditional(Conjunction('K', 'T'), Variable('J'))],
        Conditional('T', Conditional('L', 'K')),
        ['J', 'K', 'L', 'T'])

if __name__ == '__main__':
    main()





