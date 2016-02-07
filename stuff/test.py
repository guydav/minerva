def isHappy(n):
    """
    :type n: int
    :rtype: bool
    """
    visited = []
    while n != 1:
        if n in visited:
            return False

        visited.append(n)
        new_n = 0
        for digit in str(n):
            new_n += int(digit) ** 2
        print new_n
        n = new_n

    return True

print 1 <= 4 <= 6
print isHappy(4)