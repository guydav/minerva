import numpy as np

from cs152.final.counterfactual_regret import PlayableCounterfactualRegretTrainer, save_trainer, load_trainer
from cs152.final.util.bidict import bidict

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
    def __init__(self, cfr_plus=False):
        super().__init__(cfr_plus)
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

    def _history_to_human(self, chance_state, history, human_player):
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


def main():
    trainer = KuhnPokerCFRTrainer()
    trainer.train(100000, should_print_result=True)
    save_trainer(trainer, 'kuhn_trainer.pickle')

    loaded_trainer = load_trainer('kuhn_trainer.pickle')
    loaded_trainer.print()
    loaded_trainer.play(10)

    return

if __name__ == '__main__':
    main()