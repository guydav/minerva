NUM_PARAMS = 3
FEMALE_PLANT = '*'
MALE_PLANT = '#'
EMPTY = '.'


def handle_plant(marker, neighbors):
    other = (marker == FEMALE_PLANT) and MALE_PLANT or FEMALE_PLANT
    count = neighbors.count(other)
    if count < 2 or count > 3:
        return EMPTY

    return marker


def handle_empty(marker, neighbors):
    neighbors = [m for m in neighbors if m != marker]
    if 3 == len(neighbors):
        if 1 == neighbors.count(FEMALE_PLANT):
            return FEMALE_PLANT

        elif 1 == neighbors.count(MALE_PLANT):
            return MALE_PLANT

    return marker


SIMULATION_MAP = {MALE_PLANT: handle_plant, FEMALE_PLANT: handle_plant,
                  EMPTY: handle_empty}


def potential_steps(i, j, rows, cols):
    all_options = (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)
    return filter(lambda x: (x[0] >= 0) and (x[0] < rows) and
                               (x[1] >= 0) and (x[1] < cols),
                  all_options)


def simulate(h, w, t, data):
    for _ in range(t):
        new_data = [[] for _ in range(h)]
        for i in range(h):
            for j in range(w):
                current = data[i][j]
                neighbors = potential_steps(i, j, h, w)
                neighbor_values = [data[x[0]][x[1]] for x in neighbors]
                new_value = SIMULATION_MAP[current](current, neighbor_values)
                new_data[i].append(new_value)

        data = new_data

    for row in data:
        print ''.join(row)

    print


def main():
    h, w, t = map(int, raw_input().split())
    data = [raw_input() for _ in range(h)]
    simulate(h, w, t, data)


if __name__ == '__main__':
    main()