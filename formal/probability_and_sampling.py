from scipy import optimize, stats

def target_function(sigma):
    return abs(stats.norm.cdf(5.0 / sigma) - stats.norm.cdf(-3.0 / sigma) - 0.99)

def optimize():
    print optimize.minimize_scalar(target_function, (0.0000001, 2))

import random

def single_game(p_shot=0.5, streak=5, game_length=20):
    current_streak = 0

    for shot in xrange(game_length):
        if random.random() < p_shot:
            current_streak += 1

        else:
            current_streak = 0

        if current_streak == streak:
            return True

    return False

def game_iterator(n):
    for i in xrange(n):
        yield int(single_game())

def simulation(n=10**6):
    games_with_streaks = sum(game_iterator(n))
    print 'In {percentage:2.4f}% of the games there was a streak of at least 5 made shots'\
        .format(percentage=games_with_streaks * 100.0 / n)


if __name__ == '__main__':
    simulation()
