import numpy as np


# Liar die definitions
DOUBT_ACTION = 0
ACCEPT_ACTION = 1
LIAR_DIE_SIZE = 6  # try up to d20
# List of claim and response nodes


class FSICFRNode:
    def __init__(self, actions, info_set, info_set_formatter=lambda x: str(x)):
        self.actions = actions
        self.info_set = info_set
        self.info_set_formatter = info_set_formatter

        self.num_actions = len(actions)
        self.regret_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.strategy_sums = np.zeros((self.num_actions,), dtype=np.float64)

        self.utility = 0
        self.player_prob = 0
        self.opponent_prob = 0

    def get_strategy(self):
        strategy = np.max(self.regret_sums, 0)
        normalization = np.sum(strategy)

        if normalization > 0:
            strategy /= normalization

        else:
            strategy = np.ones_like(strategy) / self.num_actions

        self.strategy_sums += strategy * self.player_prob
        return strategy

    def get_average_strategy(self):
        normalization = np.sum(self.strategy_sums)

        if normalization > 0:
            return self.strategy_sums / normalization

        return np.ones_like(self.strategy_sums) / self.num_actions

    def __repr__(self):
        return '{info}: {avg}'.format(info=self.info_set_formatter(self.info_set),
                                      avg=np.array_str(self.get_average_strategy(), precision=3))


class FSICFRTrainer:
    def __init__(self, sides=LIAR_DIE_SIZE):
        self.nodes = {}
        self.sides = sides
        self.total_utility = 0

        self.response_nodes = [[FSICFRNode([ACCEPT_ACTION]) if opp_claim % sides == 0 else FSICFRNode([ACCEPT_ACTION, DOUBT_ACTION])
                                for opp_claim in range(my_claim + 1, sides + 1)]
                               for my_claim in range(sides + 1)]
        self.claim_nodes = [[FSICFRNode([range(opp_claim + 1, sides + 1)])
                                for _ in range(1, sides + 1)]
                               for opp_claim in range(sides)]

