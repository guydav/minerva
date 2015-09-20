__author__ = 'guydavidson'

'''
Generate a list of ten numbers, and sum them
'''

import random

COUNT_TO_SUM = 10
MAXIMUM_ITEM = 10

def main():
    numbers = [random.randint(0, MAXIMUM_ITEM) for i in xrange(COUNT_TO_SUM)]
    print 'The sum of %s = %d' % (str(numbers), sum(numbers))

    # alternative solutions: a for loop, reduce

if __name__ == '__main__':
    main()