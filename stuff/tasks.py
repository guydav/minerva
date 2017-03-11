def max_tasks(x, a, b, n_a, n_b):
    return min(n_a * a, n_b * b, x * n_a * n_b)


def main():
    while True:
        x, a, b, n_a, n_b = map(int, raw_input().split(','))
        print max_tasks(x, a, b, n_a, n_b)


if __name__ == '__main__':
    main()