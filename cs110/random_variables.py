from numpy import random, average


def hire_assistant(n):
    applicant_pool = random.uniform(size=n)
    current_assistant = 0
    hirings = 0

    for assistant in applicant_pool:
        if assistant > current_assistant:
            current_assistant = assistant
            hirings += 1

    return hirings


def simulate_hiring(max_n=100, trials=10):
    n_values = range(2, max_n)
    results = []

    for n in n_values:
        results.append(average([hire_assistant(n) for _ in xrange(trials)]))


def main():
    print simulate_hiring(10)


if __name__ == '__main__':
    main()



