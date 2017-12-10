import numpy as np
from collections import defaultdict, Counter
from bitarray import bitarray
from bidict import bidict
import pickle

class CounterfactualRegretNode:
    def __init__(self, actions, info_set, info_set_formatter=lambda x: str(x)):
        self.actions = actions
        self.info_set = info_set
        self.info_set_formatter = info_set_formatter

        self.num_actions = len(actions)
        self.regret_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.strategy_sums = np.zeros((self.num_actions,), dtype=np.float64)

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
        return '{info}: {avg}'.format(info=self.info_set_formatter(self.info_set),
                                      avg=np.array_str(self.get_average_strategy(), precision=3))


def save_trainer(trainer, output_path):
    with open(output_path, 'wb') as out_file:
        pickle.dump(trainer, out_file)


def load_trainer(input_path):
    with open(input_path, 'rb') as in_file:
        return pickle.load(in_file)


NUM_PLAYERS = 2


class CounterfactualRegretTrainer:
    """
    This implementation currently assumes the chance-sampling Monte Carlo formulation of CFR
    Where the chance states are sampled prior to each training iteration.

    The implementation currently only supports two-player games.

    TODO: consider implementing a chance-node supporting implementation
    """
    def __init__(self):
        self.nodes = {}
        self.total_utility = 0

    def train(self, num_iterations, num_prints=None, should_print_result=False):
        """
        The external callable for this trainer
        :param num_iterations: How many iterations should be trained
        :param num_prints: How often should a status be printed - by default 100 times over
            the entire training process, or once every 100 iterations, the smaller of the two
        :param should_print_result: Should the final average strategy be printed
        :return: None
        """
        if num_prints is None:
            num_prints = min(num_iterations // 100, 100)

        for t in range(num_iterations):
            chance_state = self._chance_sampler(t)
            initial_history = self._initial_history_generator()
            self.total_utility += self._train(chance_state, initial_history)

            if 0 == t % int(num_iterations / num_prints):
                print('After {t} iterations, average utility = {util:.3f}'.format(
                    t=t, util=self.total_utility / (t + 1)))

        if should_print_result:
            self.print()

    def print(self):
        for info_set in sorted(self.nodes):
            print(self.nodes[info_set])

    def _train(self, chance_state, history, probabilities=None, player=0):
        """
        Run a single iteration of counterfactual regret minimization learning
        :param chance_state: The chance-state for the current iteration, under
            the chance-sampling Monte Carlo formulation of CFR
        :param history: The current history of the game
        :param probabilities: The probability of arriving at the current state
            for each player
        :param player: The ordinal of the current player
        :return: The utility for the current player of the current state
        """
        opponent = 1 - player

        if probabilities is None:
            probabilities = np.ones((2,))

        utility = self._check_terminal_state(chance_state, history)
        if utility is not None:
            return utility

        info_set = self._information_set_generator(chance_state, history, player)
        if info_set not in self.nodes:
            self.nodes[info_set] = CounterfactualRegretNode(
                self._node_action_generator(chance_state, history, player),
                info_set, self._information_set_formatter_generator())

        node = self.nodes[info_set]

        # recursively CFR down the tree, returning utilities
        strategy = node.get_strategy(probabilities[player])
        utility = []
        node_utility = 0
        for action, action_prob in zip(node.actions, strategy):
            new_prob = np.copy(probabilities)
            new_prob[player] *= action_prob

            action_utility = -1 * self._train(chance_state, self._action_history_updater(history, action),
                                              new_prob, 1 - player)
            utility.append(action_utility)
            node_utility += action_utility * action_prob

        utility = np.asarray(utility)

        # compute counterfactual regret, weighted by probability of opponent arriving here
        node.regret_sums += (utility - node_utility) * probabilities[opponent]

        # return the node utility to propagate up the tree
        return node_utility

    def _chance_sampler(self, iteration=None):
        """
        Implement the chance sampler for this game
        :return: the chance state this game assumes
        """
        raise NotImplementedError()

    def _initial_history_generator(self):
        """
        Generate an initial history for the game, based on how you choose o represent it
        :return: the initial history in a format the rest of the game assumes
        """
        raise NotImplementedError()

    def _check_terminal_state(self, chance_state, history):
        """
        Check the current state of the game, between the chance state and the history,
        do determine if the game is over, and if it is, return the utility to the current player
        :param chance_state: The chance state of the game
        :param history: The history of the game
        :return: None if the game is not over, or the utility if the game is
        """
        raise NotImplementedError()

    def _information_set_generator(self, chance_state, history, player):
        """
        Map the chance state, history, and current player to the key idnetifying the information set
        :param chance_state: The chance state of the game
        :param history: The history of the game
        :param player: The ordinal of the current player
        :return: The key identifying the information set (should be hashable)
        """
        raise NotImplementedError()

    def _information_set_formatter_generator(self):
        """
        If you want to format the information sets more nicely for printing, override this
        :return: A function that maps from information set to string
        """
        return str

    def _node_action_generator(self, chance_state, history, player):
        """
        Generate the identifiers of the actions that could be taken from the node
        representing the current game state
        :param chance_state: The chance state of the game
        :param history: The history of the game
        :param player: The ordinal of the current player
        :return: An iterable of actions that can be taken from the current game state
        """
        raise NotImplementedError()

    def _action_history_updater(self, history, action):
        """
        Update the history with the latest action taken, based on how you choose
        to represent the history and the actions
        :param history: The history of the game
        :param action: The identifier for the current action taken
        :return: A new history including the latest action
        """
        raise NotImplementedError()


class PlayableCounterfactualRegretTrainer(CounterfactualRegretTrainer):
    """
    A subclass of CounterfactualRegretTrainer, supporting human-facing play.
    Requires defining several additional interface-related methods
    """
    def play(self, num_rounds=0):
        """
        The external interface to play against this trainr
        :param num_rounds: How many rounds to continue playing for, 0 to play for a long time
        :return: The overall utility for the player of all games played
        """
        if 0 == len(self.nodes):
            raise ValueError('Cannot play against untrained AI. Please train first.')

        if 0 == num_rounds:
            num_rounds = 10 ** 6

        player_utility = 0

        for t in range(1, num_rounds + 1):
            chance_state = self._chance_sampler(t)
            human_player = np.random.randint(0, NUM_PLAYERS)
            print('{player} will start this round'.format(player='You' if 0 == human_player else 'The AI'))

            player_utility += self._play(chance_state, human_player)

            print('After {t} rounds, your total utility is {util}, for an average of {avg:.3f} per round'.format(
                t=t, util=player_utility, avg=player_utility / t))
            print()

    def _play(self, chance_state, human_player):
        """
        TODO: document
        :param chance_state:
        :param human_player:
        :return:
        """
        history = self._initial_history_generator()
        player = 0
        utility = None

        print(self._chance_state_to_human(chance_state, human_player))

        while utility is None:  # while True equivalent
            human_history = self._history_to_human(history, human_player)
            if len(human_history) > 0:
                print('Game history: ' + human_history)

            utility = self._check_terminal_state(chance_state, history)
            if utility is not None:
                human_utility = utility * (1 if player == human_player else -1)
                print(self._game_over_to_human(chance_state, history, human_player, human_utility))
                return human_utility

            info_set = self._information_set_generator(chance_state, history, player)
            if info_set not in self.nodes:
                raise ValueError('Encountered unseen information set while playing human. This should never happen')

            node = self.nodes[info_set]
            action = None

            if player == human_player:
                print('Action options: ' + self._actions_to_human(node.actions))
                valid = False
                while not valid:
                    human_action = input('What action will you take? ')
                    action = self._validate_human_action_to_action(node.actions, human_action)

                    if action is not None:
                        valid = True
                    else:
                        print('Invalid action selection')

            else:
                action = np.random.choice(node.actions, p=node.get_average_strategy())
                print(self._ai_action_to_human(action))

            history = self._action_history_updater(history, action)
            player = 1 - player
            print()

        raise ValueError('This should never be reached')

    def _chance_state_to_human(self, chance_state, human_player):
        """
        Communicate the chance state to a human player
        :param chance_state: The internal representation of the chance state
        :param human_player: The ordinal of the human player
        :return: A string to be printed to the user
        """
        raise NotImplementedError()

    def _history_to_human(self, history, human_player):
        """
        Communicate the history to a human player
        :param history: The internal representation of the history
        :return: A string to be printed to the user
        """
        raise NotImplementedError()

    def _game_over_to_human(self, chance_state, history, human_player, human_utility):
        """
        Create a game-over message to the human player, informing him if he won or lost
        :param chance_state: The internal representation of the chance state
        :param history: The internal representation of the game history
        :param human_player: The ordinal of the human playEr
        :param human_utility: The utility to the human player
        :return: A string to be printed to the usre
        """
        raise NotImplementedError()

    def _actions_to_human(self, actions):
        """
        Translate the actions from the internal representation to a human friendly one
        :param actions: The actions available at the current information set
        :return: A string to be printed to the user
        """
        raise NotImplementedError()

    def _validate_human_action_to_action(self, actions, human_action):
        """
        Validate the human player's action choice and translate it into the internal representation
        :param actions: The actions available at the current information set
        :param human_action: The input from the human player
        :return: None if the action is invalid, or its proper internal representation if it is valid
        """
        raise NotImplementedError()

    def _ai_action_to_human(self, action):
        """
        Describe the AI's current action choice to the human player
        :param action: The AI's latest action choice
        :return: A string describing it to the human player
        """
        raise NotImplementedError()


# Kuhn poker related definitions
PASS = 0
BET = 1
KUHN_POKER_ACTIONS = (PASS, BET)
PASS_HISTORY = 'p'
BET_HISTORY = 'b'
KUHN_POKER_ACTION_TO_HISTORY = bidict({PASS: PASS_HISTORY, BET: BET_HISTORY})
KUHN_POKER_HISTORY_TO_HUMAN = bidict({PASS_HISTORY: 'pass', BET_HISTORY: 'bet'})
KUHN_POKER_CARDS = bidict({0: 'J', 1: 'Q', 2: 'K'})


class KuhnPokerCFRTrainer(PlayableCounterfactualRegretTrainer):
    def __init__(self):
        super().__init__()
        self.cards = np.asarray(range(3))

    def _chance_sampler(self, iteration=None):
        np.random.shuffle(self.cards)
        return self.cards

    def _check_terminal_state(self, chance_state, history):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if plays <= 1:
            return None

        player_wins = 1 if chance_state[player] > chance_state[opponent] else -1

        # double bet or double pass
        if history[-1] == history[-2]:
            double_bet = 2 if history[-1] == BET_HISTORY else 1
            return player_wins * double_bet

        # other player just passed
        if history[-1] == PASS_HISTORY:
            return 1

        return None

    def _information_set_generator(self, chance_state, history, player):
        return str(chance_state[player]) + history

    def _node_action_generator(self, chance_state, history, player):
        return KUHN_POKER_ACTIONS

    def _initial_history_generator(self):
        return ''

    def _action_history_updater(self, history, action):
        return history + KUHN_POKER_ACTION_TO_HISTORY[action]

    # PlayableCFR related definitions

    def _chance_state_to_human(self, chance_state, human_player):
        card = chance_state[human_player]
        return 'You were dealt a {card}'.format(card=KUHN_POKER_CARDS[card])

    def _history_to_human(self, history, human_player):
        human_history = ['{act} ({player})'.format(act=KUHN_POKER_HISTORY_TO_HUMAN[action],
                                                   player='you' if index % 2 == human_player else 'AI')
                         for index, action in enumerate(history)]
        return ', '.join(human_history).capitalize()

    def _game_over_to_human(self, chance_state, history, human_player, human_utility):
        return 'The AI was dealt a {ai_card} to your {card}, and you {profit} {count} chip{s}'.format(
            ai_card=KUHN_POKER_CARDS[chance_state[1 - human_player]],
            card=KUHN_POKER_CARDS[chance_state[human_player]],
            profit='gain' if human_utility > 0 else 'lose',
            count=abs(human_utility),
            s='s' if abs(human_utility) > 1 else ''
        )

    def _actions_to_human(self, actions):
        human_actions = ['{act} ({code})'.format(act=KUHN_POKER_HISTORY_TO_HUMAN[action_code],
                                                 code=action_code)
                         for action_code in [KUHN_POKER_ACTION_TO_HISTORY[action] for action in actions]]
        return ', '.join(human_actions)

    def _validate_human_action_to_action(self, actions, human_action):
        human_action = human_action.lower()

        if human_action in KUHN_POKER_HISTORY_TO_HUMAN.inverse:
            human_action = KUHN_POKER_HISTORY_TO_HUMAN.inverse[human_action][0]

        if human_action in KUHN_POKER_HISTORY_TO_HUMAN:
            action = KUHN_POKER_ACTION_TO_HISTORY.inverse[human_action][0]
            return action if action in actions else None

        # Support blank entry when there's no choice
        if 1 == len(actions) and 0 == len(human_action.strip()):
            return actions[0]

    def _ai_action_to_human(self, action):
        return 'The AI chose to {action}'.format(
            action=KUHN_POKER_HISTORY_TO_HUMAN[KUHN_POKER_ACTION_TO_HISTORY[action]])


# Dudo-related definitions

# borrowed from http://christophe-simonis-at-tiny.blogspot.kr/2008/08/python-reverse-enumerate.html
def reverse_enumerate(l):
    return zip(range(len(l) - 1, -1, -1), reversed(l))


DUDO_DEFAULT_DIE_SIZE = 6


class DudoTwoPlayerSingleDieCFRTrainer(CounterfactualRegretTrainer):
    def __init__(self, die_size=DUDO_DEFAULT_DIE_SIZE):
        super().__init__()
        self.die_size = die_size

        self.die = np.asarray(range(1, self.die_size + 1))
        self.claim_actions = NUM_PLAYERS * self.die_size
        self.num_actions = self.claim_actions + 1
        self.action_index_to_action = {i: (i // self.die_size + 1,
                                           (((i % self.die_size) + 1) % self.die_size) + 1)
                                       for i in range(self.claim_actions)}

    def _initial_history_generator(self):
        return bitarray([0] * self.num_actions)

    def _chance_sampler(self, iteration=None):
        return np.random.choice(self.die, 2, True)

    def _check_terminal_state(self, chance_state, history):
        if not history[-1]:
            return None

        # omit the last item of history to ignore the 'dudo' action
        for index, value in reverse_enumerate(history[:-1]):
            if value:
                break

        number, rank = self.action_index_to_action[index]
        counter = Counter(chance_state)
        relevant_die = counter[1]

        # 1's are wild and always count, so if the rank isn't 1 we also count occurrences of the result
        if 1 != rank:
            relevant_die += counter[rank]

        return 1 if relevant_die >= number else -1

    def _information_set_formatter_generator(self):
        def dudo_info_set_to_str(info_set):
            dice_mod = 2 ** self.num_actions
            d = info_set // dice_mod
            h = bin(info_set - (dice_mod * d))[2:]
            return '{d}, {h}'.format(d=d, h=h.rjust(self.num_actions, '0'))

        return dudo_info_set_to_str

    def _information_set_generator(self, chance_state, history, player):
        return ((2 ** self.num_actions) * chance_state[player]) + int(history.to01(), 2)

    def _node_action_generator(self, chance_state, history, player):
        # if no bets have been made, you cannot claim 'dudo'
        if not history.any():
            return range(self.claim_actions)

        # from the first unclaimed bet to 'dudo'
        for index, value in reverse_enumerate(history[:-1]):
            if value:
                break

        return range(index + 1, self.num_actions)

    def _action_history_updater(self, history, action):
        new_history = history.copy()
        new_history[action] = True
        return new_history


def main():
    # dudo_trainer = DudoTwoPlayerSingleDieCFRTrainer()
    # dudo_trainer.train(10000, should_print_result=True)

    # trainer = KuhnPokerCFRTrainer()
    # trainer.train(10000, should_print_result=True)
    # save_trainer(trainer, 'trainer.pickle')

    loaded_trainer = load_trainer('trainer.pickle')
    # loaded_trainer.print()

    loaded_trainer.play(10)


if __name__ == '__main__':
    main()

