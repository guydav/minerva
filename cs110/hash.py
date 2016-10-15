import random
import string
import timeit
from profilehooks import profile

class Crossword(object):
    def __init__(self, up_answers, across_answers):
        self.up_answers = up_answers
        self.across_answers = across_answers

        self.up_guesses = [None] * len(self.up_answers)
        self.across_guesses = [None] * len(self.across_answers)

    def guess_up(self, index, guess):
        if self.up_guesses[index] is not None:
            raise ValueError('There is already a guess for {index} up, it is {current_guess}'.format(
                index=index, current_guess=self.up_guesses[index]))

        self.up_guesses[index] = guess

    def guess_across(self, index, guess):
        if self.across_guesses[index] is not None:
            raise ValueError('There is already a guess for {index} up, it is {current_guess}'.format(
                index=index, current_guess=self.across_guesses[index]))

        self.across_guesses[index] = guess

    def clear_up_guess(self, index):
        self.up_guesses[index] = None

    def clear_across_guess(self, index):
        self.across_guesses[index] = None

"""
US Social security numbers are 7 digits long - a 10^7 space is fairly easily covered in a 32-bit address space,
about 430 times over. If there's no satellite data, it should be fine. The question of recycling also comes into play -
if SSNs are recycled sometimes, that could be a problem
"""


def random_word(length):
    return ''.join(random.choice(string.lowercase) for _ in xrange(length))


def empty_hash_table(n):
    return [[] for _ in xrange(n)]


def add_to_hash_table(hash_table, item, hash_function):
    n = len(hash_table)
    key = hash_function(item) % n
    hash_table[key].append(item)
    return hash_table


def contains(hash_table, item, hash_function):
    n = len(hash_table)
    key = hash_function(item) % n
    try:
        hash_table[key].index(item)
        return True

    except ValueError:
        return False


def remove(hash_table, item, hash_function):
    if not contains(hash_table, item, hash_function):
        raise ValueError()

    n = len(hash_table)
    key = hash_function(item) % n
    bucket = hash_table[key]
    bucket.pop(bucket.index(item))
    return hash_table


def hash_str1(key_string):
    ans = 0
    for char in key_string:
        ans += ord(char)

    return ans


def hash_str2(key_string):
    ans = 0
    for char in key_string:
        ans ^= ord(char)

    return ans


def hash_str3(key_string):
    ans = 0
    for char in key_string:
        ans = ans * 128 + ord(char)

    return ans


def hash_str4(key_string):
    random.seed(ord(key_string[0]))
    return random.getrandbits(32)


HASH_TABLE_SIZE = 5000
WORD_BANK_SIZE = 10000
WORD_LENGTH = 10


def measure_collisions(hash_table):
    collision_count = 0
    bucket_count = 0

    for bucket in hash_table:
        length = len(bucket)
        if length > 1:
            bucket_count += 1
            collision_count += (length - 1)

    return collision_count


def average_bucket_length(hash_table):
    non_empty_buckets = [bucket for bucket in hash_table if len(bucket) > 0]
    return float(sum(map(len, non_empty_buckets))) / len(non_empty_buckets)


@profile(immediate=True)
def search_word_bank(hash_table, hash_function, word_bank):
    for word in word_bank:
        contains(hash_table, word, hash_function)


def main():
    word_bank = [random_word(WORD_LENGTH) for _ in xrange(WORD_BANK_SIZE)]
    longer_word_bank = [random_word(WORD_LENGTH * 2) for _ in xrange(WORD_BANK_SIZE)]
    hash_funcs = [hash_str1, hash_str2, hash_str3, hash_str4]
    hash_tables = {hash_str1: empty_hash_table(HASH_TABLE_SIZE),
                   hash_str2: empty_hash_table(HASH_TABLE_SIZE),
                   hash_str3: empty_hash_table(HASH_TABLE_SIZE),
                   hash_str4: empty_hash_table(HASH_TABLE_SIZE)}

    for hash_func in hash_funcs:
        table = hash_tables[hash_func]
        for word in word_bank:
            add_to_hash_table(table, word, hash_func)

        collisions = measure_collisions(table)
        bucket_length = average_bucket_length(table)
        print 'Table with function {func} had {col} collisions and average bucket length of {length}'.format(
            func=hash_func.func_name, col=collisions, length=bucket_length)

        search_word_bank(table, hash_func, word_bank)
        search_word_bank(table, hash_func, longer_word_bank)


if __name__ == '__main__':
    main()