from scipy import stats
import numpy as np
# import matplotlib.pyplot as plot
# %matplotlib inline


DEFAULT_HORIZONTAL_SIZE = 12
DEFAULT_VERTICAL_SIZE = 8
DEFAULT_DPI = 800
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0


class GameTheorySimulation(object):
    def __init__(self, agent_factory, agent_count, num_rounds, initial_proportion=None):
        self.agent_factory = agent_factory
        self.agent_count = agent_count
        self.num_rounds = num_rounds
        self.initial_proportion = initial_proportion
        self.results = []

        self.agents = [self.agent_factory() for i in xrange(self.agent_count)]

    def simulate(self):
        previous_proportion = self.initial_proportion
        for round in xrange(self.num_rounds):
            sell_count = [agent.simulate(previous_proportion) for agent in self.agents].count(True)
            self.results.append(sell_count)
            previous_proportion = float(sell_count) / self.agent_count

    # def graph(self):
    #     plot.figure(figsize=(DEFAULT_HORIZONTAL_SIZE, DEFAULT_VERTICAL_SIZE),
    #             dpi=DEFAULT_DPI)
    #
    #     # +1 to allow to separately graph the initial proportion, if exists
    #     x_values = np.arange(0, self.num_rounds + 1)
    #     plot.plot(x_values, self.results, label='Stores Selling GMO Products')
    #
    #     plot.legend(loc='best')
    #     plot.xlabel('Week / Round')
    #     plot.ylabel('Count of Stores Selling')
    #     plot.grid(True)
    #     plot.show()


class SimulationAgent(object):
    def __init__(self, threshold_proportion):
        self.threshold_proportion = np.clip(threshold_proportion, MIN_THRESHOLD, MAX_THRESHOLD)

    def simulate(self, previous_proportion):
        # If no previous proportion given, randomize - week 1
        if previous_proportion is None:
            previous_proportion = np.random.uniform(MIN_THRESHOLD, MAX_THRESHOLD)

        return previous_proportion >= self.threshold_proportion


class ConstantThresholdAgentFactory(object):
    def __init__(self, threshold_proportion):
        self.threshold_proportion = threshold_proportion

    def __call__(self, *args, **kwargs):
        return SimulationAgent(self.threshold_proportion)


class DistributionAgentFactory(object):
    def __init__(self, distribution, **sample_parameters):
        self.distribution = distribution
        self.sample_parameters = sample_parameters

    def __call__(self, *args, **kwargs):
        return SimulationAgent(self.distribution(**self.sample_parameters))


def main():
    # sim = GameTheorySimulation(ConstantThresholdAgentFactory(0.5), 100, 20)
    # sim = GameTheorySimulation(DistributionAgentFactory(np.random.normal, loc=0.5, scale=0.2), 100, 20)
    # sim = GameTheorySimulation(DistributionAgentFactory(np.random.uniform, low=0, high=1), 100, 20)
    sim = GameTheorySimulation(DistributionAgentFactory(np.random.b, low=0, high=1), 100, 20)
    sim.simulate()
    print sim.results

main()