'''
General gameplan:
    1. Create table to store the transition matrix.
    2. Execute the following stream of queries:
        a. Select all games form 2015.
        b. From each game, select all at-bats.
        c. For each at-bat longer than one pitch, log the pitches.

'''

import peewee
from pitchfx_models import *
from datetime import datetime


# -- Helper Methods -- #

def select_games(start_date=None, end_date=None):
    query = Game.select()

    if start_date:
        query = query.where(Game.date >= start_date)

    if end_date:
        query = query.where(Game.date <= end_date)

    return query.order_by(Game.date.asc())


def main():
    games = select_games(datetime(2015, 1, 1))
    for game in games:
        print len(game.at_bats)


if __name__ == '__main__':
    main()