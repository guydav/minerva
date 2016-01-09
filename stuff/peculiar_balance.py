def answer(x):
    '''
    x: the weight on the left side to be balanced
    return: a list of where to place each weight - "L" / "-" / "R"
    '''
    # create a list of all weights to be used:
    if x <= 0:
        return

    current_weight = 1
    weights = [current_weight]

    while sum(weights) < x:
        current_weight *= 3
        weights.append(current_weight)

    solution_weight = 0
    target_weight = x

    solution = []

    while solution_weight != target_weight:
        current_weight = weights.pop()
        diff = abs(target_weight - solution_weight)

        if weights and diff <= sum(weights):
            solution.append('-')
            continue

        diff_weight_on_right = abs(target_weight - solution_weight - current_weight)
        if diff_weight_on_right < diff:
            # weight should go on the right
            solution.append('R')
            solution_weight += current_weight

        else:
            # weight goes on left
            solution.append('L')
            target_weight += current_weight

    for remaining_weight in weights:
        # handle the remaining weights well
        solution.append('-')

    solution.reverse()
    return solution

if __name__ == '__main__':
    for x in xrange(1000):
        answer(x)