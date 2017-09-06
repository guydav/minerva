
def shortz(n, count=0, total=0, cache={}):
    while n >= 1:
        total += n
        count += 1
        if n in cache:
            n = cache[n]

        elif n % 2 == 0:
            n >>= 2

        else:
            n = 3 * n + 1

        #if not (count % 100):
        print count, n

    return count, total, cache

c, _, cache = shortz(98325112)
_, s, cache = shortz(329376974512194, cache=cache)
print c * s