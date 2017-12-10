import numpy as np
import itertools


# Constant definitions
ROCK = 0
PAPER = 1
SCISSORS = 2
ROCK_PAPER_SCISSORS_ACTIONS = (ROCK, PAPER, SCISSORS)


class RegretMinimizationTrainer:
    def __init__(self, actions, utility_func):
        self.actions = actions
        self.utility_func = utility_func

        self.num_actions = len(actions)
        self.regret_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.strategy_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.last_action = None

    def get_strategy(self):
        strategy = np.where(self.regret_sums > 0, self.regret_sums, 0)
        normalization = np.sum(strategy)

        if normalization > 0:
            strategy /= normalization

        else:
            strategy = np.ones_like(strategy) / self.num_actions

        self.strategy_sums += strategy
        return strategy

    def get_action(self):
        action_index = np.random.choice(range(len(self.actions)), p=self.get_strategy())
        self.last_action = self.actions[action_index]
        return self.last_action

    def update_regret(self, opponent_action):
        utility = self.utility_func(self.last_action, opponent_action)
        self.regret_sums += [self.utility_func(alt_action, opponent_action) - utility
                             for alt_action in self.actions]

    def get_average_strategy(self):
        normalization = np.sum(self.strategy_sums)

        if normalization > 0:
            return self.strategy_sums / normalization

        return np.ones_like(self.strategy_sums) / self.num_actions


def rock_paper_scissors_utility(action, opponent_action):
    utility = (action - opponent_action) % 3
    if utility == 2:
        utility = -1
    return utility


def train_rock_paper_scissors(t_max, opponent_strategy):
    trainer = RegretMinimizationTrainer(ROCK_PAPER_SCISSORS_ACTIONS, rock_paper_scissors_utility)

    for t in range(t_max):
        action = trainer.get_action()
        opponent_action = np.random.choice(ROCK_PAPER_SCISSORS_ACTIONS, p=opponent_strategy)
        trainer.update_regret(opponent_action)

        if 0 == (t % 1000):
            print(trainer.get_average_strategy())

    return trainer.get_average_strategy()


def train_both_rock_paper_scissors(t_max):
    trainer = RegretMinimizationTrainer(ROCK_PAPER_SCISSORS_ACTIONS, rock_paper_scissors_utility)
    opp_trainer = RegretMinimizationTrainer(ROCK_PAPER_SCISSORS_ACTIONS, rock_paper_scissors_utility)

    for t in range(t_max):
        action = trainer.get_action()
        opponent_action = opp_trainer.get_action()

        trainer.update_regret(opponent_action)
        opp_trainer.update_regret(action)

        if 0 == (t % 1000):
            print(trainer.get_average_strategy(), opp_trainer.get_average_strategy())

    return trainer.get_average_strategy(), opp_trainer.get_average_strategy()


def generate_battlefield_actions(num_soldiers, num_battles):
    """
    This reasonably naive code assumes under 56 battles (A-Za-z) for each one
    :param num_soldiers: the total number of soldiers that could be allocated
    :param num_battles: the total number of battles to allocate soldiers to
    :return: the different possible action allocations
    """
    battles = [chr(ord('A') + x) for x in range(num_battles)]
    combinations = itertools.combinations_with_replacement(battles, num_soldiers)
    actions = [np.asarray([comb.count(b) for b in battles])
               for comb in combinations]
    return actions


def battlefield_utility(action, opponent_action):
    return np.sum(np.sign(action - opponent_action))


def train_battlefield_against_uniform(t_max, num_soldiers, num_battlers, max_actions_print=10):
    actions = np.asarray(generate_battlefield_actions(num_soldiers, num_battlers))
    trainer = RegretMinimizationTrainer(actions, battlefield_utility)

    for t in range(t_max):
        action = trainer.get_action()
        opponent_action = np.random.choice(action)
        trainer.update_regret(opponent_action)

        if 0 == (t % 1000) and 0 != t:
            average_strategy = trainer.get_average_strategy()
            print(actions[np.argpartition(average_strategy, -max_actions_print)[-max_actions_print:]])

    return trainer.get_average_strategy()


def train_battlefield_against_uniform(t_max, num_soldiers, num_battlers, max_actions_print=10):
    actions = np.asarray(generate_battlefield_actions(num_soldiers, num_battlers))
    trainer = RegretMinimizationTrainer(actions, battlefield_utility)
    opp_trainer = RegretMinimizationTrainer(actions, battlefield_utility)

    for t in range(t_max):
        action = trainer.get_action()
        opponent_action = opp_trainer.get_action()

        trainer.update_regret(opponent_action)
        opp_trainer.update_regret(action)

        if 0 == (t % 1000) and 0 != t:
            average_strategy = trainer.get_average_strategy()
            print(actions[np.argpartition(average_strategy, -max_actions_print)[-max_actions_print:]])

    return trainer.get_average_strategy()


if __name__ == '__main__':
    # train_rock_paper_scissors(10 ** 5, [0.4, 0.3, 0.3])
    # train_both_rock_paper_scissors(10 ** 5)
    # print(generate_battlefield_actions(10, 4))
    # train_battlefield_against_uniform(10000, 10, 4)
    train_battlefield_against_uniform(10 ** 5, 10, 4)