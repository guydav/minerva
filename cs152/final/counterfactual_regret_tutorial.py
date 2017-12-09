import numpy as np
from collections import defaultdict, Counter
from bitarray import bitarray

PASS = 0
BET = 1
KUHN_POKER_ACTIONS = (PASS, BET)
PASS_HISTORY = 'p'
BET_HISTORY = 'b'
KUHN_POKER_ACTION_TO_HISTORY = {
    PASS: PASS_HISTORY,
    BET: BET_HISTORY,
}


class CounterfactualRegretNode:
    def __init__(self, actions, utility_func, info_set):
        self.actions = actions
        self.utility_func = utility_func
        self.info_set = info_set

        self.num_actions = len(actions)
        self.regret_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.strategy_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.last_action = None

    def get_strategy(self, realization_weight):
        strategy = np.where(self.regret_sums > 0, self.regret_sums, 0)
        normalization = np.sum(strategy)

        if normalization > 0:
            strategy /= normalization

        else:
            strategy = np.ones_like(strategy) / self.num_actions

        self.strategy_sums += strategy * realization_weight
        return strategy

    def get_average_strategy(self):
        normalization = np.sum(self.strategy_sums)

        if normalization > 0:
            return self.strategy_sums / normalization

        return np.ones_like(self.strategy_sums) / self.num_actions

    def __repr__(self):
        return '{info}: {avg}'.format(info=self.info_set,
                                      avg=np.array_str(self.get_average_strategy(), precision=3))


def kuhn_poker_check_terminal_state(chance_state, history):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if plays <= 1:
        return False, None

    player_wins = 1 if chance_state[player] > chance_state[opponent] else -1

    # double bet or double pass
    if history[-1] == history[-2]:
        double_bet = 2 if history[-1] == BET_HISTORY else 1
        return True, player_wins * double_bet

    # other player just passed
    if history[-1] == PASS_HISTORY:
        return True, 1

    return False, None


def cfr(nodes, chance_state, history, probabilities=None, player=0,
        info_set_generator=lambda chance, player, history: str(chance[player]) + history,
        check_terminal_state=kuhn_poker_check_terminal_state,
        node_action_generator=lambda x: KUHN_POKER_ACTIONS,
        action_history_updater=lambda hist, act: hist + KUHN_POKER_ACTION_TO_HISTORY[act]):
    """
    The tutorial implementation of CFR, before I begin cleaning it up
    Assming this is always a two-player game.
    :param nodes: The mapping between information sets and their node representations
    :param chance_state: The cards, where cards[0] is the card of player0, and cards[1] is card of player1
    :param history: The moves up until this point, as a string
    :param probabilities: The probabilities of arriving to this point for each player
    :param player: the current player
    :param info_set_generator: A function that generates the key determining the information set of a current state
    :param check_terminal_state: A function that checks if a state is terminal, and returns the utility if it is
    :param node_action_generator: A function mapping from the history of a node to the actions that can be taken
    :param action_history_updater: A function that updates the history according to the action chosen
    :return: The utility of this node
    """
    opponent = 1 - player

    if probabilities is None:
        probabilities = np.ones((2,))

    terminal, utility = check_terminal_state(chance_state, history)
    if terminal:
        return utility

    info_set = info_set_generator(chance_state, player, history)
    if info_set not in nodes:
        nodes[info_set] = CounterfactualRegretNode(node_action_generator(history), None, info_set)

    node = nodes[info_set]

    # recursively CFR down the tree, returning utilities
    strategy = node.get_strategy(probabilities[player])
    utility = []
    node_utility = 0
    for action, action_prob in zip(node.actions, strategy):
        new_prob = np.copy(probabilities)
        new_prob[player] *= action_prob

        action_utility = -1 * cfr(nodes, chance_state, action_history_updater(history, action), new_prob,
                                  1 - player, info_set_generator, check_terminal_state,
                                  node_action_generator, action_history_updater)
        utility.append(action_utility)
        node_utility += action_utility * action_prob

    utility = np.asarray(utility)

    # compute counterfactual regret, weighted by probability of opponent arriving here
    node.regret_sums += (utility - node_utility) * probabilities[opponent]

    # return the node utility to propogate up the tree
    return node_utility


def cfr_trainer(num_iterations, num_prints=100):
    nodes = {}
    cards = np.asarray(range(3))
    total_utility = 0

    for t in range(num_iterations):
        np.random.shuffle(cards)
        total_utility += cfr(nodes, cards, '')

        if 0 == t % int(num_iterations / num_prints) and t > 0:
            print('After {t} iterations, average utility = {util:.3f}'.format(
                t=t, util=total_utility / t))

    for info_set in sorted(nodes):
        print(nodes[info_set])


# Dudo-related definitions

DUDE_NUM_PLAYERS = 2
DUDO_DICE_SIZE = 6
DUDO_CLAIM_ACTIONS = DUDO_DICE_SIZE * DUDE_NUM_PLAYERS
DUDO_NUM_ACTIONS = DUDO_CLAIM_ACTIONS + 1  # the 'dudo' action
DUDO_ACTION_INDEX_TO_ACTION = {i: (i // DUDO_DICE_SIZE + 1,
                                   (((i % DUDO_DICE_SIZE) + 1) % DUDO_DICE_SIZE) + 1)
                               for i in range(DUDO_NUM_ACTIONS - 1)}


# borrowed from http://christophe-simonis-at-tiny.blogspot.kr/2008/08/python-reverse-enumerate.html
def reverse_enumerate(l):
    return zip(range(len(l)-1, -1, -1), reversed(l))


def dudo_terminal_state(chance_state, history):
    if not history[-1]:
        return False, None

    # omit the last item of history to ignore the 'dudo' action
    for index, value in reverse_enumerate(history[:-1]):
        if value:
            break

    number, rank = DUDO_ACTION_INDEX_TO_ACTION[index]
    counter = Counter(chance_state)
    relevant_die = counter[1]

    # 1's are wild and always count, so if the rank isn't 1 we also count occurrences of the result
    if 1 != rank:
        relevant_die += counter[rank]

    return True, 1 if relevant_die >= number else -1


def dudo_info_set_generator(chance_state, player, history):
    return ((2 ** DUDO_NUM_ACTIONS) * chance_state[player]) + int(history.to01(), 2)


def dudo_node_action_generator(history):
    # if no bets have been made, you cannot claim 'dudo'
    if not history.any():
        return range(DUDO_CLAIM_ACTIONS)

    # from the first unclaimed bet to 'dudo'
    for index, value in reverse_enumerate(history[:-1]):
        if value:
            break

    return range(index + 1, DUDO_NUM_ACTIONS)


def dudo_action_history_updater(history, action):
    new_history = history.copy()
    new_history[action] = True
    return new_history


def dudo_info_set_to_str(info_set):
    dice_mod = 2 ** DUDO_NUM_ACTIONS
    d = info_set // dice_mod
    h = bin(info_set - (dice_mod * d))[2:]
    return '{d}, {h}'.format(d=d, h=h.rjust(DUDO_NUM_ACTIONS, '0'))


def dudo_cfr_trainer(num_iterations, num_prints=100):
    nodes = {}
    die = np.asarray(range(1, DUDO_DICE_SIZE + 1))
    total_utility = 0

    for t in range(num_iterations):
        die_rolls = np.random.choice(die, 2, True)
        history = bitarray([0] * DUDO_NUM_ACTIONS)
        total_utility += cfr(nodes, die_rolls, history,
                             check_terminal_state=dudo_terminal_state,
                             info_set_generator=dudo_info_set_generator,
                             node_action_generator=dudo_node_action_generator,
                             action_history_updater=dudo_action_history_updater)

        if 0 == t % int(num_iterations / num_prints):
            print('After {t} iterations, average utility = {util:.3f}'.format(
                t=t, util=total_utility / (t + 1)))

    for info_set in sorted(nodes):
        print(dudo_info_set_to_str(info_set), nodes[info_set].get_average_strategy())


if __name__ == '__main__':
    # cfr_trainer(100000)
    dudo_cfr_trainer(1000, 100)
