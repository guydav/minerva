import matplotlib.pyplot as plot

# -- Functions -- #


def generate_prey_rate_function(a, b):
    """
    Generates a Lotka-Volterra prey function:
    dx/dt = alpha * x - beta * x * y
    :param a: Alpha, the prey population growth parameter, > 0
    :param b: Beta, the prey-predator interaction prey death parameter, > 0
    :return: A function for the rate of change of prey population
    """
    def prey_rate_function(x, y, t):
        return a * x - b * x * y

    return prey_rate_function


def generate_predator_rate_function(c, d):
    """
    Generates a Lotka-Volterra predator function:
    dy/dt = delta * x * y - gamma * y
    :param c: Gamma, the predator population death parameter, > 0
    :param d: Delta, the prey-predator interaction predator growth parameter, > 0
    :return: A function for the rate of change of predator population
    """
    def predator_rate_function(x, y, t):
        return d * x * y - c * y

    return predator_rate_function


def eulers_method(rates, initial_values, time_values, time_step, min_values=(0, 0)):
    """
    Estimate a set of differential equations using Euler's method of linear estimation
    :param rates: An iterable with functions representing the rates of change of the different variable. Each function
    must accept as many parameters as they are rates + the time parameter -
    so if there are two functions (rates of change of x and y), each much accept three parameters (x, y, t)
    :param initial_values: An iterable with the initial values for each quantity represented - (x_0, y_0)
    :param time_values: A slice-able with the time values to estimate for
    :param time_step: h, the time step to take in each round
    :param min_values: An slice-able with minimal values each variable can take - defaults to
    :return: The sets of values each variable takes
    """
    # t = 0
    values = [[val] for val in initial_values]
    previous_values = list(initial_values)

    # next steps
    for t in time_values[1:]:
        params = previous_values + [t]
        rates_of_change = [rate_func(*params) for rate_func in rates]

        for i in xrange(len(values)):
            next_value = max(previous_values[i] + time_step * rates_of_change[i], min_values[i])
            values[i].append(next_value)
            previous_values[i] = next_value

    return values


def eulers_method_lotka_volterra(initial_prey, initial_predator, a, b, c, d, t_max, t_step):
    """
    Simulates the Lotka-Volterra equations using Euler's Method, and graphs the result
    :param initial_prey: The initial prey population (x_0)
    :param initial_predator: The inigial predator population (y_0)
    :param a: Alpha, the prey population growth parameter, > 0
    :param b: Beta, the prey-predator interaction prey death parameter, > 0
    :param c: Gamma, the predator population death parameter, > 0
    :param d: Delta, the prey-predator interaction predator growth parameter, > 0
    :param t_max: The maximal time value to simulate until, > 0
    :param t_step: The time step to take, > 0
    :return: None - graphs result using matplotlib
    """
    prey_rate = generate_prey_rate_function(a, b)
    pred_rate = generate_predator_rate_function(c, d)

    time_values = [t * t_step for t in range(0, int(t_max / t_step))]

    prey_values, predator_values = \
        eulers_method((prey_rate, pred_rate), (initial_prey, initial_predator), time_values, t_step)

    figure, axes = plot.subplots()

    axes.plot(time_values, prey_values, label='Prey Population', color='green')
    axes.plot(time_values, predator_values, label='Predator Population', color='red')
    axes.set_xlabel('Time')
    axes.set_ylabel('Population Sizes')
    axes.legend(loc='best')
    axes.grid(True)

    plot.show()

# -- Main -- #

INITIAL_PREY = 100
INITIAL_PREDATOR = 100
ALPHA = 1
BETA = 0.02
GAMMA = 1
DELTA = 0.03
TIME_MAX = 100
TIME_STEP = 0.01

def main():
    eulers_method_lotka_volterra(INITIAL_PREY, INITIAL_PREDATOR, ALPHA, BETA, GAMMA, DELTA, TIME_MAX, TIME_STEP)


if __name__ == '__main__':
    main()
