#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################
# Speech recognition #
######################
# 1. Work through the following resources:
#     Viterbi algorithm. (n.d.). In Wikipedia. Retrieved November 9, 2016, from
#     https://en.wikipedia.org/wiki/Viterbi_algorithm
#     The 44 Phonemes in English. (n.d.). Retrieved November 9, 2016, from
#     http://www.dyslexia-reading-well.com/44-phonemes-in-english.html
# 2. Read and run the code given below.
"""
# 3. Answer the following questions:
#   a. What does the transition_probability table describe?
The probabilities of moving from one hidden state to the next one
#   b. What does the emission_probability table describe?
The probabilities of observing a given phoneme from a certain hidden state
#   c. What does the start_probability table describe?
Some initial probability - probably related to overall language prevalance
#   d. What does the Viterbi algorithm do with these tables?
Computes the most likely hidden state path transition - in this case, the
observed states are the phonemes heard, and the hidden states are letters,
making this presumably a speech-to-text model
#   e. Describe the optimal substructure found in this problem.
The transition to get to the current state is a product of the transition
to get to the previous state * previous state for current state - making
it easily contstructable bottom-up
#   f. How should one interpret the output of the Viterbi algorithm?
The most likely sequence of hidden states to have generated the observed
state; in this case, the most likely sequence of letters spoken to
generate the observed / heard / recorded phonemes
"""


def viterbi(obs, states, start_p, transition_p, emit_p):
    V = [{}]

    for state in states:
        V[0][state] = {"prob": start_p[state] * emit_p[state][obs[0]], "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})

        for state in states:
            max_tr_prob = 0

            for previous_state in states:
                current_transition_p = \
                    V[t - 1][previous_state]["prob"] * transition_p[previous_state][state]

                if max_tr_prob < current_transition_p:
                    max_prob = current_transition_p * emit_p[state][obs[t]]
                    max_tr_prob = current_transition_p
                    V[t][state] = {"prob": max_prob, "prev": previous_state}

    for line in dptable(V):
        print line

    output = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            output.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        output.append(V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    output.reverse()
    print 'The steps of states are ' + ' '.join(
        output) + ' with highest probability of %s' % max_prob


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%9s" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.9s: " % state + " ".join("%.9s" % ("%.3E" % v[state]["prob"])
                                          for v in V)


states = 'abcdr'
observations = ('/a/', '/b/', '/r/', '/ã/', '/k/', '/a/', '/d/', '/a/', '/b/',
                '/r/', '/ã/')
start_probability = {'a': 0.4, 'b': 0.1, 'c': 0.1, 'd': 0.3, 'r': 0.1}

transition_probability = {'a': {'a': 0,
                                'b': 0.3,
                                'c': 0.3,
                                'd': 0.3,
                                'r': 0.1},
                          'b': {'a': 0.2,
                                'b': 0,
                                'c': 0.2,
                                'd': 0.1,
                                'r': 0.5},
                          'c': {'a': 0.5,
                                'b': 0.1,
                                'c': 0.1,
                                'd': 0.1,
                                'r': 0.1},
                          'd': {'a': 0.5,
                                'b': 0.2,
                                'c': 0.2,
                                'd': 0.0,
                                'r': 0.1},
                          'r': {'a': 0.7,
                                'b': 0.1,
                                'c': 0.1,
                                'd': 0.1,
                                'r': 0}}
emission_probability = {'a': {'/a/': 0.4,
                              '/ã/': 0.4,
                              '/b/': 0.05,
                              '/r/': 0.05,
                              '/d/': 0.05,
                              '/k/': 0.05},
                        'b': {'/a/': 0.05,
                              '/ã/': 0.05,
                              '/b/': 0.65,
                              '/r/': 0.05,
                              '/d/': 0.2,
                              '/k/': 0.05},
                        'c': {'/a/': 0.05,
                              '/ã/': 0.05,
                              '/b/': 0.05,
                              '/r/': 0.05,
                              '/d/': 0.05,
                              '/k/': 0.75},
                        'd': {'/a/': 0.05,
                              '/ã/': 0.05,
                              '/b/': 0.2,
                              '/r/': 0.05,
                              '/d/': 0.6,
                              '/k/': 0.05},
                        'r': {'/a/': 0.05,
                              '/ã/': 0.05,
                              '/b/': 0.05,
                              '/r/': 0.7,
                              '/d/': 0.05,
                              '/k/': 0.1}}

viterbi(observations, states, start_probability, transition_probability,
        emission_probability)

