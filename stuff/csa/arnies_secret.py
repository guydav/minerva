import re


PATTERN = '(ZERO|ONE)(!+)'
TEST_DATA = """ZERO! ZERO! ONE! ZERO!! ZERO!!! ZERO!!! ONE! ONE!! ZERO! ONE! ZERO! ZERO! ONE!!! ONE! ZERO!! ONE!!! ZERO!! ONE!!! ZERO!! ONE!! ONE!!! ZERO! ZERO!!! ONE!! ZERO! ZERO!!! ONE!! ZERO! ZERO!! ZERO!!! ZERO!!! ZERO!! ZERO! ONE! ZERO!!! ZERO!!! ZERO! ZERO!!! ZERO!!! ONE!! ZERO!!! ONE!!! ZERO!!! ONE! ZERO! ZERO! ONE!! ZERO!!! ZERO! ONE! ZERO! ZERO!! ONE!! ONE!!! ONE!! ZERO!! ZERO!!! ONE!!! ZERO!!! ZERO!! ONE! ONE!! ONE!!! ONE!!! ZERO!! ONE! ZERO!! ZERO! ONE! ONE!! ZERO!!! ZERO!! ZERO!!! ONE! ZERO!!! ZERO! ZERO!!! ONE!! ZERO!!! ZERO! ZERO! ONE!!! ZERO!! ZERO!!! ZERO! ZERO!! ONE! ONE!!! ZERO!! ZERO!!! ONE!! ZERO! ZERO! ZERO!!! ZERO!!! ZERO! ZERO!! ONE!! ZERO! ZERO!!! ZERO!!! ZERO!!! ONE! ONE!!! ZERO! ONE! ZERO!!! ZERO!!! ONE! ONE! ONE! ONE! ZERO!! ONE!"""
TRANSLATION = {'ZERO': '0', 'ONE': '1' }

if __name__ == '__main__':
    pattern = re.compile(PATTERN)
    with open('./arnies_secret.txt') as data_file:
        data = data_file.read()

        bin_string = ''.join([TRANSLATION[match.group(1)] * len(match.group(2))
                              for match in pattern.finditer(data)])
        print bin_string
        print len(bin_string)

