DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

def answer(population, x, y, strength):
    '''
    population: a 2D array of rabbits and their strengths
    x: the X coordinate of patient Z
    y: the Y coordinate of patient Z
    strength: the strength of the virus
    return: a copy of the population array, with -1 for infected Rabbits
    '''
    # create a copy of the array
    infected_population = []
    for row in population:
        infected_population.append(row[:])

    # handle the edge case
    if infected_population[x][y] > strength:
        return infected_population

    frontier = [(x,y)]
    checked = set()

    while frontier:
        current = frontier.pop()
        checked.add(current)

        curr_x, curr_y = current

        if infected_population[curr_x][curr_y] > strength:
            continue

        # infected
        infected_population[curr_x][curr_y] = -1

        for direction in DIRECTIONS:
            new_rabbit = (curr_x + direction[0], curr_y + direction[1])
            # check it's a valid rabit location
            if (not new_rabbit in checked) and \
                (0 <= new_rabbit[0] < len(population)) and \
                (0 <= new_rabbit[1] < len(population[new_rabbit[0]])):
                frontier.append(new_rabbit)

    return infected_population


if __name__ == '__main__':
    # print answer([[1, 2, 3], [2, 3, 4], [3, 2, 1]], 0, 0, 2)
    print answer([[6, 7, 2, 7, 6], [6, 3, 1, 4, 7], [0, 2, 4, 1, 10], [8, 1, 1, 4, 9], [8, 7, 4, 9, 9]], 2, 1, 5)