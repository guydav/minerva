"""
General gameplan:
    1. Create table to store the transition matrix.
    2. Execute the following stream of queries:
        a. Select all games form 2015.
        b. From each game, select all at-bats.
        c. For each at-bat longer than one pitch, log the pitches.

"""
from pitchfx_models import *
from datetime import datetime
import progressbar


# -- Helper Methods -- #


def select_games(start_date=None, end_date=None):
    query = Game.select()

    if start_date:
        query = query.where(Game.date >= start_date)

    if end_date:
        query = query.where(Game.date <= end_date)

    return query.order_by(Game.date.asc())


def process_at_bat(at_bat):
    pitches = at_bat.pitches

    # -1 as we do not want to cont the last pitch - also implicitly acts as an if len(pitches) > 1
    for i in xrange(len(pitches) - 1):
        first = pitches[i]
        second = pitches[i+1]

        transition = PitchTransition.create(
            at_bat=at_bat, batter=at_bat.batter, pitcher=at_bat.pitcher,
            first_ball=first.ball, first_des=first.des, first=first.id,
            first_on_1b=first.on_1b, first_on_2b=first.on_2b, first_on_3b=first.on_3b,
            first_pitch=first, first_pitch_type=first.pitch_type, first_strike=first.strike,
            first_sv=first.sv, first_type=first.type, first_type_confidence=first.type_confidence,
            second_ball=second.ball, second_des=second.des, second=second.id,
            second_on_1b=second.on_1b, second_on_2b=second.on_2b, second_on_3b=second.on_3b,
            second_pitch=second, second_pitch_type=second.pitch_type, second_strike=second.strike,
            second_sv=second.sv, second_type=second.type, second_type_confidence=second.type_confidence)
        transition.save()

    database.commit()


def process_game(game):
    at_bats = game.at_bats
    for at_bat in at_bats:
        process_at_bat(at_bat)


def process_2015_season():
    games = select_games(datetime(2015, 1, 1))
    progress = progressbar.ProgressBar()
    for game in progress(games):
        process_game(game)


def add_pitchers_and_batters_to_transitions():
    empty_transitions = PitchTransition.select().where(PitchTransition.batter == 0).\
        where(PitchTransition.pitcher == 0)

    print 'Found {len} transitions with batters and pitchers'.format(len=len(empty_transitions))

    progress = progressbar.ProgressBar()

    for pt in progress(empty_transitions):
        pt.batter = pt.at_bat.batter
        pt.pitcher = pt.at_bat.pitcher
        pt.save()

    database.commit()


def main():
    # process_2015_season()
    add_pitchers_and_batters_to_transitions()
    # pass


if __name__ == '__main__':
    main()
