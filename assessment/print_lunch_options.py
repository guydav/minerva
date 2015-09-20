__author__ = 'guydavidson'

LUNCH_OPTIONS = ['Hamburger', 'Cheeseburger', 'Schnitzel', 'Hummcus', 'Stir-fry', 'Fried Rice',
                 'Salad', 'Pizza', 'Dim Sum', 'Soup Dumplings', 'Cake', 'Leftovers']
COUNT_TO_PRINT = 5
PRINT_FORMAT = '%d) %s'

import random

def main():
    for i in xrange(COUNT_TO_PRINT):
        choice = random.choice(LUNCH_OPTIONS)
        print PRINT_FORMAT % (i+1, choice)
        LUNCH_OPTIONS.remove(choice)

if __name__ == '__main__':
    main()