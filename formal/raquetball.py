import random
import tabulate
from utils import *


SERVE_WIN_PROBABILITY = 0.6
RETURN_WIN_PROBABILITY = 0.5
PLAY_TO = 21
START_SERVE = True
NUM_TRIALS = 10000


def single_game(p_serve=SERVE_WIN_PROBABILITY, p_return=RETURN_WIN_PROBABILITY, play_to=PLAY_TO,
                start_serve=START_SERVE):
    my_score = 0
    opponent_score = 0
    current_serve = start_serve

    while my_score < play_to and opponent_score < play_to:
        point = random.random()

        if (current_serve and point < p_serve) or (not current_serve and point < p_return):
            my_score += 1
            current_serve = True

        else:
            opponent_score += 1
            current_serve = False

    return int(my_score == play_to)


def trial_generator(num_trials=NUM_TRIALS):
    for i in xrange(num_trials):
        yield single_game()


def run_simulation():
    my_total = sum(trial_generator())
    print 'The simulated winning percentage is {p:.3}%'.format(p=float(my_total)*100/NUM_TRIALS)


def build_table(p_win=SERVE_WIN_PROBABILITY, p_loss=RETURN_WIN_PROBABILITY, play_to=PLAY_TO,
                start_serve=START_SERVE):

    result = []
    for r in xrange(play_to + 1):
        row = []
        result.append(row)

        for c in xrange(play_to + 1):
            row.append({True: 0, False: 0})

    result[0][0][True] = 1
    result[0][0][False] = 1

    frontier = PriorityQueue()
    visited = set()
    frontier.put((1, 0), 0)
    frontier.put((0, 1), 0)

    while frontier:
        current = frontier.get()
        if current in visited:
            continue

        visited.add(current)
        row, col = current

        p_current_win = 0
        p_current_loss = 0

        if row == 0:
            # first row, can only win to get here
            p_current_win = result[row][col -1][True] * p_win

        elif col == 0:
            # first column, can only lose to get here
            if row == 1:
                # First loss, edge-case for starting on serve
                p_current_loss = result[row - 1][col][False] * (1 - p_win)

            else:
                p_current_loss = result[row - 1][col][False] * (1 - p_loss)

        elif row == play_to and col == play_to:
            # The score 21-21 is impossible
            pass

        elif col == play_to:
            # last column, can only win to get here
            p_current_win = result[row][col -1][True] * p_win
            p_current_win += result[row][col -1][False] * p_loss

        elif row == play_to:
            # last row, can only lose to get here
            p_current_loss = result[row - 1][col][True] * (1 - p_win)
            p_current_loss += result[row - 1][col][False] * (1 - p_loss)

        else:
            # coming from a win
            p_current_win = result[row][col -1][True] * p_win
            p_current_win += result[row][col -1][False] * p_loss

            # coming from a loss
            p_current_loss = result[row - 1][col][True] * (1 - p_win)
            p_current_loss += result[row - 1][col][False] * (1 - p_loss)

        result[row][col][True] = p_current_win
        result[row][col][False] = p_current_loss

        if row < play_to:
            next_row = (row + 1, col)
            frontier.put(next_row, sum(next_row))

        if col < play_to:
            next_col = (row, col + 1)
            frontier.put(next_col, sum(next_col))

    return result


def print_table():
    table = build_table()

    table_to_print = []
    row_num = 0
    for row in table:
        table_to_print.append([row_num] + ['{p_w:.4f}|{p_l:.4f}'.format(p_w=col[True], p_l=col[False]) for col in row])
        row_num += 1

    header_row = ['***'] + range(22)
    print tabulate.tabulate(table_to_print, headers=header_row)


    total_win_p = sum([row[21][True] for row in table])
    total_loss_p = sum([col[False] for col in table[21]])

    print 'The total win probability is {p:.4f}'.format(p=total_win_p)
    print 'The total loss probability is {p:.4f}'.format(p=total_loss_p)
    print 'The overall win probability is {p:.4f}'.format(p=(total_win_p / (total_win_p + total_loss_p)))
    print 'The overall loss probability is {p:.4f}'.format(p=(total_loss_p / (total_win_p + total_loss_p)))



def main():
    print_table()



if __name__ == '__main__':
    main()