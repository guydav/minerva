__author__ = 'guydavidson'

'''
Prints function values with a certain increment on a certain range
'''

import math

START = 0
END = 2 * math.pi
FACTOR = 0.1
SPECIAL_CASES = {0.5:0}
PRINT_FORMAT = '%.2f:\t%.2f'


def foo(x):
    return x ** 2 + 3

'''
using factor so I can use xrange for non-integer steps
'''
def iterate_and_print(start, end, function , factor = 1, special_cases = []):
    # correct the end-point according to the factor
    end = int(end / factor)

    # add 1, end-point is exclusive
    for x in xrange(start, end + 1):
        factored_x = x * factor

        if factored_x in special_cases:
            result = special_cases[factored_x]

        else:
            result = function(factored_x)

        print PRINT_FORMAT % (factored_x, result)

def main():
    iterate_and_print(START, END, foo, FACTOR, SPECIAL_CASES)

if __name__ == '__main__':
    main()