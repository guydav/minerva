import math

# def answer(x, y, n):
#     if x + y > n + 1:
#         return 0
#
#     if x == 1 and y == 1:
#         if n == 1:
#             return 1
#
#         return 0
#
#     if x == 1:
#         return one_side_count(y, n - 1)
#
#     if y == 1:
#         return one_side_count(x, n -1)
#
#     total = 0
#     for max_index in xrange(x - 1, n - y + 1):
#         total += two_side_count(x, y, n, max_index)
#         # total += one_side_count(x, max_index) + one_side_count(y, n - max_index)
#
#     return total
#
# def two_side_count(l, r, n, max_index):
#     return math.factorial(n - l - r + 2) * choose(max_index, l - 2) * choose(n - max_index - 1, r - 2)
#
# def one_side_count(x, n):
#     return choose(n - 1, x - 2) * math.factorial(n - x + 1)
#
# def choose(n, k):
#     return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
#
# print answer(2, 2, 3)

class State(object):
    def __init__(self, max_index, remaning_l, remaning_r, n, state=None):
        self.max_index = max_index
        self.l = remaning_l
        self.r = remaning_r
        self.n = n

        if state:
            self.state = state
        else:
            self.state = [0] * n
            self.state[max_index] = 1


    def is_terminal(self):
        return self.l + self.r == 0


    def options(self):
        return math.factorial(self.state.count(0))


    def __repr__(self):
        return 'L: {l} R: {r} [{state}]'.format(l=self.l, r=self.r, state=' '.join([str(x) for x in self.state]))


    def next_states(self):
        if self.is_terminal():
            return

        new_states = []

        if self.l:
            iter_max = self.max_index
            if self.l == 1:
                iter_max = 1


            for i in xrange(iter_max):
                if self.state[i] == 0:
                    new_state = self.state[:]
                    new_state[i] = max(self.state) + 1
                    new_states.append(State(self.max_index, self.l - 1, self.r, self.n, new_state))

        if self.r:
            iter_min = self.max_index + 1
            if self.r == 1:
                iter_min = self.n - 1


            for i in xrange(iter_min, self.n):
                if self.state[i] == 0:
                    new_state = self.state[:]
                    new_state[i] = max(self.state) + 1
                    new_states.append(State(self.max_index, self.l, self.r - 1, self.n, new_state))

        return new_states

def answer(l, r, n):
    if l + r > n + 1:
        return 0

    if l == 1 and r == 1:
        if n == 1:
            return 1

        return 0

    max_indices = []

    if l == 1:
        max_indices.append(0)

    elif r == 1:
        max_indices.append(n - 1)

    else:
        max_indices += range(l - 1, n - r + 1)

    options = 0
    states = []
    for max_index in max_indices:
        states.append(State(max_index, l - 1, r - 1, n))

    while states:
        current_state = states.pop()

        if current_state.is_terminal():
            options += current_state.options()

        else:
            states.extend(current_state.next_states())

    return options

print answer(3, 5, 10)
