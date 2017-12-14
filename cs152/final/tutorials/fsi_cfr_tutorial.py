import numpy as np
from tabulate import tabulate


# Liar die definitions
DOUBT_ACTION = 0
ACCEPT_ACTION = 1

LIAR_DIE_SIZE = 6  # try up to d20
# List of claim and response nodes


class FSICFRNode:
    def __init__(self, actions, info_set='', info_set_formatter=lambda x: str(x)):
        self.actions = actions
        self.info_set = info_set
        self.info_set_formatter = info_set_formatter

        self.num_actions = len(actions)
        self.regret_sums = np.zeros((self.num_actions,), dtype=np.float64)
        self.strategy_sums = np.zeros((self.num_actions,), dtype=np.float64)

        self.utility = 0
        self.player_prob = 0
        self.opponent_prob = 0
        self.current_strategy = None

    def get_strategy(self):
        strategy = np.where(self.regret_sums > 0, self.regret_sums, 0)
        normalization = np.sum(strategy)

        if normalization > 0:
            strategy /= normalization

        else:
            strategy = np.ones_like(strategy) / self.num_actions

        self.strategy_sums += strategy * self.player_prob
        self.current_strategy = strategy
        return strategy

    def get_average_strategy(self):
        normalization = np.sum(self.strategy_sums)

        if normalization > 0:
            return self.strategy_sums / normalization

        return np.ones_like(self.strategy_sums) / self.num_actions

    def __repr__(self):
        return '{info}: {avg}'.format(info=self.info_set_formatter(self.info_set),
                                      avg=np.array_str(self.get_average_strategy(), precision=3))


class FSICFRLieDieTrainer:
    def __init__(self, sides=LIAR_DIE_SIZE):
        self.sides = sides

        self.response_nodes = [[FSICFRNode([ACCEPT_ACTION]) if opp_claim % sides == 0 else FSICFRNode([DOUBT_ACTION, ACCEPT_ACTION])
                                for opp_claim in range(my_claim + 1, sides + 1)]
                               for my_claim in range(sides + 1)]
        self.claim_nodes = [[FSICFRNode(list(range(opp_claim + 1, sides + 1)))
                             for _ in range(1, sides + 1)]
                            for opp_claim in range(sides)]

    def print(self):
        # print initial claim policy for each roll
        initial_roll_strategies = [(initial_roll,
                                    np.array_str(self.claim_nodes[0][initial_roll - 1].get_average_strategy(), precision=3))
                                   for initial_roll in range(1, self.sides + 1)]
        print(tabulate(initial_roll_strategies, ('Initial Roll', 'Strategy'), 'fancy_grid'))

        # print response policy following each previous two claims
        response_strategies = [[(my_claim,
                                 opp_claim,
                                 np.array_str(self.response_nodes[my_claim][opp_claim - my_claim - 1].get_average_strategy(), precision=3))
                                for opp_claim in range(my_claim + 1, self.sides + 1)]
                               for my_claim in range(self.sides + 1)]
        flat_response_strategies = [item for sublist in response_strategies for item in sublist]
        print(tabulate(flat_response_strategies,
                       ('Previous Player Claim', 'Opponent Claim', 'Strategy'),
                       'fancy_grid'))

        # print claim policy following each previous claim and roll result
        claim_strategies = [[(opp_claim,
                              roll,
                              np.array_str(self.claim_nodes[opp_claim][roll - 1].get_average_strategy(), precision=3))
                             for roll in range(1, self.sides + 1)]
                            for opp_claim in range(self.sides)]

        flat_claim_strategies = [item for sublist in claim_strategies for item in sublist]
        print(tabulate(flat_claim_strategies,
                       ('Opponent Claim', 'Roll', 'Strategy'),
                       'fancy_grid'))

    def train(self, num_iterations, num_prints=None, should_print_result=False):
        """
        The external callable for this trainer
        :param num_iterations: How many iterations should be trained
        :param num_prints: How often should a status be printed - by default 100 times over
            the entire training process, or once every 100 iterations, the smaller of the two
        :param should_print_result: Should the final average strategy be printed
        :return: None
        """
        num_iterations = int(num_iterations)

        if num_prints is None:
            num_prints = min(num_iterations // 100, 100)

        for t in range(num_iterations):
            # the chance state
            roll_after_accepted_claims = [np.random.randint(1, self.sides + 1) for _ in range(self.sides)]
            self._train(roll_after_accepted_claims)

            # reset strategy sums half way through training
            if num_iterations // 2 == t:
                for node_list in self.response_nodes:
                    for node in node_list:
                        node.strategy_sums *= 0

                for node_list in self.claim_nodes:
                    for node in node_list:
                        node.strategy_sums *= 0

            if 0 == t % int(num_iterations / num_prints) and t > 0:
                print('Finished {t} iterations'.format(t=t))

        if should_print_result:
            self.print()

    def _train(self, roll_after_accepted_claims):
        initial_node = self.claim_nodes[0][roll_after_accepted_claims[0] - 1]
        initial_node.player_prob = 1
        initial_node.opponent_prob = 1

        regret = [0] * self.sides

        for opponent_claim in range(self.sides + 1):
            # visit response nodes
            if opponent_claim > 0:
                for previous_player_claim in range(opponent_claim):
                    node = self.response_nodes[previous_player_claim][opponent_claim - previous_player_claim - 1]
                    strategy = node.get_strategy()
                    accept_index = node.actions.index(ACCEPT_ACTION)

                    if opponent_claim < self.sides:
                        next_node = self.claim_nodes[opponent_claim][roll_after_accepted_claims[opponent_claim] - 1]
                        next_node.player_prob += node.opponent_prob

                        next_node.opponent_prob += strategy[accept_index] * node.player_prob

            # visit claim nodes
            if opponent_claim < self.sides:
                node = self.claim_nodes[opponent_claim][roll_after_accepted_claims[opponent_claim] - 1]
                strategy = node.get_strategy()

                for player_claim in range(opponent_claim + 1, self.sides + 1):
                    claim_action_index = player_claim - opponent_claim - 1
                    player_claim_prob = strategy[claim_action_index]
                    if player_claim_prob > 0:
                        next_node = self.response_nodes[opponent_claim][claim_action_index]
                        next_node.player_prob += node.opponent_prob
                        next_node.opponent_prob += player_claim_prob * node.player_prob

        # backpropogate utilities, updating strategies
        for opponent_claim in reversed(range(0, self.sides + 1)):
            # visit claim nodes backward
            if opponent_claim < self.sides:
                node = self.claim_nodes[opponent_claim][roll_after_accepted_claims[opponent_claim] - 1]
                strategy = node.current_strategy
                node.utility = 0

                for player_claim in range(opponent_claim + 1, self.sides + 1):
                    claim_action_index = player_claim - opponent_claim - 1
                    next_node = self.response_nodes[opponent_claim][claim_action_index]

                    child_utility = -1 * next_node.utility
                    regret[claim_action_index] = child_utility
                    node.utility += strategy[claim_action_index] * child_utility

                for index, action_prob in enumerate(strategy):
                    regret[index] -= node.utility
                    node.regret_sums[index] += node.opponent_prob * regret[index]

                node.player_prob = 0
                node.opponent_prob = 0

            # visit response node backward
            if opponent_claim > 0:
                for previous_player_claim in range(opponent_claim):
                    node = self.response_nodes[previous_player_claim][opponent_claim - previous_player_claim - 1]
                    strategy = node.current_strategy
                    node.utility = 0

                    doubt_utility = 1 if opponent_claim > roll_after_accepted_claims[previous_player_claim] else -1
                    regret[DOUBT_ACTION] = doubt_utility

                    node.utility += strategy[DOUBT_ACTION] * doubt_utility

                    if opponent_claim < self.sides:
                        next_node = self.claim_nodes[opponent_claim][roll_after_accepted_claims[opponent_claim] - 1]
                        regret[ACCEPT_ACTION] = next_node.utility
                        node.utility += strategy[ACCEPT_ACTION] * next_node.utility

                    for index, action_prob in enumerate(strategy):
                        regret[index] -= node.utility
                        node.regret_sums[index] += node.opponent_prob * regret[index]

                    node.player_prob = 0
                    node.opponent_prob = 0


def main():
    trainer = FSICFRLieDieTrainer()
    trainer.train(1e5)
    trainer.print()


if __name__ == '__main__':
    main()

