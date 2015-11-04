__author__ = 'guydavidson'

# -- Imports -- #
import random

# -- Constants -- #

DEFAULT_LEARNING_CONSTANT = 0.1
BIAS = 1
SHOULD_BIAS = False


# -- Classes -- #

class Perceptron(object):
    # def __init__(self, weights, bias=True, previous=None, dest=None, learning_constant=DEFAULT_LEARNING_CONSTANT):
    def __init__(self, weights, previous=None, dest=None, learning_constant=DEFAULT_LEARNING_CONSTANT):
        if type(weights) is int:
            self.weights = [0] * weights

        elif weights:
            self.weights = weights
        #
        # self.bias = bias
        # if bias:
        #     self.weights.append(0)

        self.learning_constant = learning_constant

        if previous:
            self.previous = previous
            for p in previous:
                p.dest = self

        else:
            self.previous = []

        self.dest = dest

    def activate(self, result):
        if self.dest:
            return result

        else:
            return int(result > 0)

    def feed_forward(self, inputs):
        # if not (len(inputs) + (1 if self.bias else 0)) == len(self.weights):
        if not len(inputs) == len(self.weights):
            raise ValueError('Inputs {inputs} do not match weights {weights}. Aborting...'
                             .format(inputs=inputs, weights=self.weights))

        if self.previous:
            inputs = [prev.feed_forward(inputs) for prev in self.previous]
            # Re-adding in the bias
            if SHOULD_BIAS:
                inputs.append(BIAS)

        # if self.bias:
        #     # Assignment instead of append or += so the bias doesn't propagate
        #     inputs = inputs + [1]

        # print self.weights, inputs

        return self.activate(sum(map(lambda w, i: w * i, self.weights, inputs)))

    def train(self, inputs, expected_result):
        actual_result = self.feed_forward(inputs)
        error = expected_result - actual_result
        self.correct(inputs, error)

    def correct(self, inputs, error, chain_contribution=1):
        # Chain rule:
        if self.previous:
            for (prev, weight) in zip(self.previous, self.weights):
                prev.correct(inputs, error, weight * chain_contribution)

        # print self.weights, inputs, error, chain_contribution, self.learning_constant

        self.weights = map(lambda w, i: w + (i * error * chain_contribution * self.learning_constant), self.weights,
                           inputs)
        # print expected_result, actual_result, error, self.weights


# -- Main -- #

def train_perceptron(perceptron, learning_function, input_size=2, min_input=0, max_input=1, bias=False, training_length=100000):
    data = [[random.randint(min_input, max_input) for i in xrange(input_size)] for j in xrange(training_length)]
    if bias:
        [d.append(BIAS) for d in data]

    for d in data:
        perceptron.train(d, learning_function(*d[:2]))


def create_and_train_perceptron(learning_function, size=2, min_input=0, max_input=1, bias=True):
    p = Perceptron(size + (1 if bias else 0))
    train_perceptron(p, learning_function, size, min_input, max_input, bias)
    return p


def main():
    # and_perceptron = create_and_train_perceptron(lambda a, b: a and b)
    # print 'AND', and_perceptron.weights
    #
    # nand_perceptron = create_and_train_perceptron(lambda a, b: int(not (a and b)))
    # print 'NAND', nand_perceptron.weights
    #
    # or_perceptron = create_and_train_perceptron(lambda a, b: a or b)
    # print 'OR', or_perceptron.weights
    #
    # nor_perceptron = create_and_train_perceptron(lambda a, b: int(not (a or b)))
    # print 'NOR', nor_perceptron.weights
    #
    # y_x_perceptron = create_and_train_perceptron(lambda x, y: int(y > x), 2, 0, 100)
    # print 'Y > X', y_x_perceptron.weights

    f_perceptron = Perceptron(2)
    g_perceptron = Perceptron(2)
    h_perceptron = Perceptron(2, previous=[f_perceptron, g_perceptron])
    train_perceptron(h_perceptron, lambda x, y: x ^ y, bias=SHOULD_BIAS)

    for p in (f_perceptron, g_perceptron, h_perceptron):
        print p.weights

    for i in xrange(2):
        for j in xrange(2):
            print i, j, h_perceptron.feed_forward([i, j])


if __name__ == '__main__':
    main()
