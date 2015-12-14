"""
General gameplan:
    1. Create table to store the transition matrix.
    2. Execute the following stream of queries:
        a. Select all games form 2015.
        b. From each game, select all at-bats.
        c. For each at-bat longer than one pitch, log the pitches.

"""
from pitchfx_models import *
import tabulate
import numpy


# -- Helper Methods -- #


def print_independent_probabilities(query_filter=None):
    pitch_counts = {}

    for pitch_type in PitchType.select():
        current_type = pitch_type.id
        count_query = Pitch.select().where(Pitch.pitch_type == current_type)

        if query_filter:
            count_query.where(query_filter)

        pitch_counts[current_type] = count_query.count()

    # TODO: Think of a better way to handle the None (unclassified) pitches
    pitch_counts[None] = Pitch.select().where(Pitch.pitch_type is None).count()

    total = sum(pitch_counts.values())
    probabilities = {pt: (float(pitch_counts[pt]) / total) for pt in pitch_counts}

    ordered_keys = sorted(pitch_counts.keys(), key=pitch_counts.get, reverse=True)
    rows = []

    joint_probabilities = {}

    for first_pitch_type in ordered_keys:
        row = [first_pitch_type]
        joint_probabilities[first_pitch_type] = {}

        for second_pitch_type in ordered_keys:
            row.append(probabilities[second_pitch_type])
            joint_probabilities[first_pitch_type][second_pitch_type] = \
                probabilities[first_pitch_type] * probabilities[second_pitch_type]

        rows.append(row)

    title = ['*****'] + ordered_keys
    print tabulate.tabulate(rows, title)

    return ordered_keys, joint_probabilities


def print_conditional_probabilities(ordered_keys, query_filter=None):
    transition_counts = {}
    pitch_types = [pt.id for pt in PitchType.select()] + [None]

    for first_pitch_type in pitch_types:
        transition_counts[first_pitch_type] = {}

        for second_pitch_type in pitch_types:
            count_query = PitchTransition.select().where(
                PitchTransition.first_pitch_type == first_pitch_type).where(
                PitchTransition.second_pitch_type == second_pitch_type)

            if query_filter:
                count_query.where(query_filter)

            transition_counts[first_pitch_type][second_pitch_type] = count_query.count()

    first_pitch_totals = {first_pitch_type: sum(transition_counts[first_pitch_type].values())
                          for first_pitch_type in pitch_types}
    first_pitch_total = sum(first_pitch_totals.values())

    rows = []
    joint_probabilities = {}

    for first_pitch_type in ordered_keys:
        first_pitch_transitions = transition_counts[first_pitch_type]
        row_total = sum(first_pitch_transitions.values())
        joint_probabilities[first_pitch_type] = {}
        first_pitch_type_probability = float(first_pitch_totals[first_pitch_type]) / first_pitch_total

        row = [first_pitch_type]
        for second_pitch_type in ordered_keys:
            second_pitch_conditional_probability = float(first_pitch_transitions[second_pitch_type]) / row_total
            row.append(second_pitch_conditional_probability)

            joint_probabilities[first_pitch_type][second_pitch_type] = \
                first_pitch_type_probability * second_pitch_conditional_probability

        rows.append(row)

    title = ['*****'] + ordered_keys
    print tabulate.tabulate(rows, title)
    return joint_probabilities


def entropy_from_probability_matrix(matrix):
    """
    H(X,Y) = Sum over x in Ax, y in Ay: P(x,y) * log2(1/P(x,y)
    :param matrix: 2d dict (dictionary of dictionaries)
    :return:
    """
    joint_entropy = 0
    for x in matrix:
        current = matrix[x]
        for y in current:
            p_x_y = current[y]
            if p_x_y:
                joint_entropy += (p_x_y * numpy.log2(1.0 / p_x_y))

    return joint_entropy


def kullback_leibler_divergence(p, q):
    """
    D_KL(P||Q) = Sum over over x: P(x) * log2(P(x) / Q(x))
    :param p: 2d dict (dictionary of dictionaries)
    :param q: 2d dict (dictionary of dictionaries) with the same structure as P
    :return:
    """
    joint_entropy = 0
    for x in p:
        p_x = p[x]
        q_x = q[x]

        for y in p_x:
            p_x_y = p_x[y]
            q_x_y = q_x[y]

            if p_x_y and q_x_y:
                joint_entropy += (p_x_y * numpy.log2(p_x_y / q_x_y))

    return joint_entropy


def print_entropies(independent_joint_probabilities, conditional_joint_probabilities):
    print 'Independent H(X,Y) = {h}'.format(h=entropy_from_probability_matrix(independent_joint_probabilities))
    print 'Conditional H(X,Y) = {h}'.format(h=entropy_from_probability_matrix(conditional_joint_probabilities))
    print 'D_KL(Independent, Conditional) = {d_kl}'\
        .format(d_kl=kullback_leibler_divergence(independent_joint_probabilities, conditional_joint_probabilities))
    print 'D_KL(Conditional, Independent) = {d_kl}'\
        .format(d_kl=kullback_leibler_divergence(conditional_joint_probabilities, independent_joint_probabilities))


def calculate_probabilities(independent_filter=None, conditional_filter=None):
    ordered_keys, independent_joint_probabilities = print_independent_probabilities(independent_filter)
    conditional_joint_probabilities = print_conditional_probabilities(ordered_keys, conditional_filter)
    return independent_joint_probabilities, conditional_joint_probabilities


def print_for_pitcher(pitcher_id):
    pitcher = Player.get(Player.eliasid == pitcher_id)
    independent_joint_probabilities, conditional_joint_probabilities = calculate_probabilities(
        Pitch.at_bat.pitcher == pitcher, PitchTransition.at_bat.pitcher == pitcher)
    print_entropies(independent_joint_probabilities, conditional_joint_probabilities)


def print_for_everyone():
    independent_joint_probabilities, conditional_joint_probabilities = calculate_probabilities()
    print_entropies(independent_joint_probabilities, conditional_joint_probabilities)


def main():
    print_for_pitcher(477132)


if __name__ == '__main__':
    main()
