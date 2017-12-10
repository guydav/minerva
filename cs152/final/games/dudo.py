from cs152.final.counterfactual_regret import CounterfactualRegretTrainer, NUM_PLAYERS
import numpy as np
from collections import Counter
from bitarray import bitarray


def reverse_enumerate(l):
    """
    # borrowed from
    http://christophe-simonis-at-tiny.blogspot.kr/2008/08/python-reverse-enumerate.html
    :param l: The iterable to provide a reversed enumerate over
    :return: An enumerate-like over the input, in reversed order
    """
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
    dudo_trainer = DudoTwoPlayerSingleDieCFRTrainer()
    dudo_trainer.train(1000, should_print_result=True)

    return


if __name__ == '__main__':
    main()