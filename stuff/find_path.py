NUM_PARAMS = 3
START_VALUE = 1


def potential_steps(i, j, rows, cols):
    all_options = (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)
    return filter(lambda x: (x[0] >= 0) and (x[0] < rows) and
                               (x[1] >= 0) and (x[1] < cols),
                  all_options)


def find_path(n, rows, cols, data):
    i = 0
    j = 0
    current = 1

    for i in range(rows):
        if START_VALUE in data[i]:
            j = data[i].index(START_VALUE)
            break

    while n != data[i][j]:
        steps = potential_steps(i, j, rows, cols)
        found = False

        if 0 == len(steps):
            return False

        for new_i, new_j in steps:
            if current + 1 == data[new_i][new_j]:
                i = new_i
                j = new_j
                current += 1
                found = True
                break

        if not found:
            return False

    return True


def main():
    num_cases = int(raw_input())
    raw_input()

    for _ in range(num_cases):
        n, rows, cols = [int(raw_input()) for _ in range(NUM_PARAMS)]
        data = [map(int, raw_input().split(' ')) for _ in range(rows)]
        raw_input()
        print str(find_path(n, rows, cols, data)).lower()

    print


if __name__ == '__main__':
    main()
