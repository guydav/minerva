import numpy as np

from cs152.final.counterfactual_regret import PlayableCounterfactualRegretTrainer, NUM_PLAYERS, save_trainer, load_trainer
from cs152.final.util.bidict import bidict

# Kuhn poker related definitions
PASS = 0
BET = 1
AD_POKER_ACTIONS = (PASS, BET)
PASS_HISTORY = 'p'
BET_HISTORY = 'b'
DEAL_HISTORY = '|'
AD_POKER_ACTION_TO_HISTORY = bidict({PASS: PASS_HISTORY, BET: BET_HISTORY})
AD_POKER_HISTORY_TO_HUMAN = bidict({PASS_HISTORY: 'pass', BET_HISTORY: 'bet', DEAL_HISTORY: 'dealt'})
AD_POKER_CARDS = bidict({0: '2', 1: 'A'})
AD_POKER_NUM_ROUNDS = 2


class AceDeucePokerCFRTrainer(PlayableCounterfactualRegretTrainer):
    def __init__(self, cfr_plus=False):
        super().__init__(cfr_plus)
        self.cards = np.asarray(range(len(AD_POKER_CARDS)))

    def _chance_sampler(self, iteration=None):
        return np.random.choice(self.cards, (NUM_PLAYERS, AD_POKER_NUM_ROUNDS), True)

    def _inject_chance_into_history(self, history, chance_state):
        if '' == history:
            return DEAL_HISTORY

        if 2 == history.count(DEAL_HISTORY):
            return history

        round_status = self._round_over(None, history)
        if round_status is None or round_status != 0:
            return history

        return history + DEAL_HISTORY

    def _round_over(self, chance_state, history):
        # TODO: consider caching this function
        bet_multiplier = 1 + history.count(BET_HISTORY) // 2
        history = history[history.rfind(DEAL_HISTORY) + 1:]
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if plays <= 1:
            return None

        player_wins = self._compare_hands(chance_state, player, opponent) if chance_state is not None else 0
        # double bet or double pass
        if history[-1] == history[-2]:
            return player_wins * bet_multiplier

        # other player just passed
        if history[-1] == PASS_HISTORY:
            return bet_multiplier

        return None

    def _compare_hands(self, chance_state, player, opponent):
        if chance_state is None:
            return 0

        player_pair = chance_state[player][0] == chance_state[player][1]
        opponent_pair = chance_state[opponent][0] == chance_state[opponent][1]

        # Return 1 if player has a higher pair, -1 if opponent has a higher pair, 0 if tied
        if player_pair and opponent_pair:
            p = chance_state[player][0]
            o = chance_state[opponent][0]
            return (p > o) - (p < o)

        if player_pair:
            return 1

        if opponent_pair:
            return -1

        # both have an A-2
        return 0

    def _check_terminal_state(self, chance_state, history):
        if history.endswith('|'):
            return None

        return self._round_over(chance_state, history)

    def _information_set_generator(self, chance_state, history, player):
        return str(chance_state[player][:history.count(DEAL_HISTORY)]) + history

    def _node_action_generator(self, chance_state, history, player):
        return AD_POKER_ACTIONS

    def _initial_history_generator(self):
        return ''

    def _action_history_updater(self, history, action):
        return history + AD_POKER_ACTION_TO_HISTORY[action]

    # PlayableCFR related definitions

    def _chance_state_to_human(self, chance_state, human_player):
        # Printing the history handles it in this case
        return ''

    def _history_to_human(self, chance_state, history, human_player):
        human_history = []
        card_index = 0

        for index, action in enumerate(history):
            if action == DEAL_HISTORY:
                info = AD_POKER_CARDS[chance_state[human_player][card_index]]
                card_index += 1

            else:
                info = 'you' if index % 2 == ((human_player + history.count(DEAL_HISTORY)) % 2) else 'AI'

            human_history.append('{act} ({info})'.format(act=AD_POKER_HISTORY_TO_HUMAN[action], info=info))

        out = ', '.join(human_history)
        return out[0].upper() + out[1:]

    def _game_over_to_human(self, chance_state, history, human_player, human_utility):
        return 'The AI was dealt a {ai_cards} to your {cards}, and you {profit} {count} chip{s}'.format(
            ai_cards=''.join([AD_POKER_CARDS[c] for c in chance_state[1 - human_player]]),
            cards=''.join([AD_POKER_CARDS[c] for c in chance_state[human_player]]),
            profit='gain' if human_utility > 0 else 'lose',
            count=abs(human_utility),
            s='' if abs(human_utility) == 1 else 's'
        )

    def _actions_to_human(self, actions):
        human_actions = ['{act} ({code})'.format(act=AD_POKER_HISTORY_TO_HUMAN[action_code],
                                                 code=action_code)
                         for action_code in [AD_POKER_ACTION_TO_HISTORY[action] for action in actions]]
        return ', '.join(human_actions)

    def _validate_human_action_to_action(self, actions, human_action):
        human_action = human_action.lower()

        if human_action in AD_POKER_HISTORY_TO_HUMAN.inverse:
            human_action = AD_POKER_HISTORY_TO_HUMAN.inverse[human_action][0]

        if human_action in AD_POKER_HISTORY_TO_HUMAN:
            action = AD_POKER_ACTION_TO_HISTORY.inverse[human_action][0]
            return action if action in actions else None

        # Support blank entry when there's no choice
        if 1 == len(actions) and 0 == len(human_action.strip()):
            return actions[0]

    def _ai_action_to_human(self, action):
        return 'The AI chose to {action}'.format(
            action=AD_POKER_HISTORY_TO_HUMAN[AD_POKER_ACTION_TO_HISTORY[action]])


def main():
    trainer = AceDeucePokerCFRTrainer(cfr_plus=True)
    trainer.train(100000, should_print_result=True)
    # save_trainer(trainer, 'ad_trainer.pickle')

    # loaded_trainer = load_trainer('ad_trainer.pickle')
    # loaded_trainer.play(10)

    return

if __name__ == '__main__':
    main()