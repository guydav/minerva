# Return True if the given string contains an appearance of "xyz"
# where the xyz is not directly preceeded by a period (.).
# So "xxyz" counts but "x.xyz" does not.
#
# xyz_there('abcxyz')  True
# xyz_there('abc.xyz')  False
# xyz_there('xyz.abc')  True


def xyz_there(string):
    for index in range(len(string) - 2):
        if (index == 0 or string[index-1] != '.') and \
            string[index:index+3] == 'xyz':

            return True

    return False

# print xyz_there('abcxyz')
# print xyz_there('abc.xyz')
# print xyz_there('xyz.abc')
# print xyz_there('...xyz.abc')

def is_palindrome(string):
    length = len(string)
    for index in range(length / 2):
        if string[index] != string[length - 1 - index]:
            return False

    return True

def one_line_palin(string):
    return string == string[::-1]


def other_xyz(string):
    index = string.find('xyz')
    if index == 0:
        return True

    while index != -1:
        if string[index - 1] != '.':
            return True
        else:
            index = string.find('xyz', index + 1)

    return False

print xyz_there('xyz.xyz')

# s = 'abc.xyz'
# print s.find('xyz')
# print s.find('c.x')
# print s.find('Colette')


# print one_line_palin('Jakob')
# print one_line_palin('tacocat')
# print one_line_palin('boob')
