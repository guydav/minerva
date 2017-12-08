import numpy as np
from collections import defaultdict

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


def check_terminal_state(cards, history, plays, player, opponent):
    if plays <= 1:
        return False, None

    player_wins = 1 if cards[player] > cards[opponent] else -1

    # double bet or double pass
    if history[-1] == history[-2]:
        double_bet = 2 if history[-1] == BET_HISTORY else 1
        return True, player_wins * double_bet

    # other player just passed
    if history[-1] == PASS_HISTORY:
        return True, 1

    return False, None


def cfr(nodes, cards, history='', probabilities=None,
        actions=KUHN_POKER_ACTIONS,
        action_to_history=KUHN_POKER_ACTION_TO_HISTORY):
    """
    The tutorial implementation of CFR, before I begin cleaning it up
    Assming this is always a two-player game.
    :param nodes: The mapping between information sets and their node representations
    :param cards: The cards, where cards[0] is the card of player0, and cards[1] is card of player1
    :param history: The moves up until this point, as a string
    :param probabilities: The probabilities of arriving to this point for each player
    :param actions: The actions that can be taken in this game
        TODO: switch this to a node-based representation
    :param action_to_history: The mapping from an action to its historical representation
        TODO: represent better
    :return: The utility of this node
    """
    if probabilities is None:
        probabilities = np.ones((2,))

    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    terminal, utility = check_terminal_state(cards, history, plays, player, opponent)
    if terminal:
        return utility

    # we define the information set as the card this player was dealt
    # as well as all visible moves
    info_set = str(cards[player]) + history
    if info_set not in nodes:
        nodes[info_set] = CounterfactualRegretNode(actions, None, info_set)

    node = nodes[info_set]

    # recursively CFR down the tree, returning utilities
    strategy = node.get_strategy(probabilities[player])
    utility = []
    node_utility = 0
    for action, action_prob in zip(node.actions, strategy):
        new_prob = np.copy(probabilities)
        new_prob[player] *= action_prob

        action_utility = -1 * cfr(nodes, cards, history + action_to_history[action], new_prob)
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
        total_utility += cfr(nodes, cards)

        if 0 == t % int(num_iterations / num_prints) and t > 0:
            print('After {t} iterations, average utility = {util:.3f}'.format(
                t=t, util=total_utility / t))

    for info_set in sorted(nodes):
        print(nodes[info_set])

if __name__ == '__main__':
    cfr_trainer(100000)

