import numpy as np
from collections import defaultdict
from functools import partial

ONE_OVER_SQRT_TWO = 1.0 / (2 ** 0.5)
DEFAULT_TANGLED_PAIR = np.array((ONE_OVER_SQRT_TWO, ONE_OVER_SQRT_TWO))
DEFAULT_NUM_BITS = 2 ** 12
COMPUTATIONAL_VECS = (np.matrix((1, 0)),
                      np.matrix((0, 1)))
TSIRELSON_CORRELATION_BOUND = -2 * (2 ** 0.5)
DEFAULT_NOISE_MU = 0
DEFAULT_NOISE_SIGMA = 0.05
NOISE_SIZE = (1, 2)
ZERO_NOISE_LAMBDA = lambda: np.array((0, 0))


def normalize(vec):
    return vec / np.linalg.norm(vec)


class QuantumBasis(object):
    def __init__(self, basis_vectors, name, noise=ZERO_NOISE_LAMBDA):
        self.basis_vectors = basis_vectors
        self.name = name
        self.noise = noise

    def measure(self, qubit):
        projections = [abs(vec.dot(qubit.T))[0, 0]
                       for vec in self.basis_vectors]
        projections = np.squeeze(normalize(projections + self.noise()))
        probabilities = [p ** 2 for p in projections]
        return np.random.choice(2, p=probabilities)

    def interpret(self, bit_value):
        return self.basis_vectors[bit_value]

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)

    def __cmp__(self, other):
        return cmp(self.name, other.name)


class QubitPair(object):
    def __init__(self, initial_state=DEFAULT_TANGLED_PAIR,
                 noise=ZERO_NOISE_LAMBDA):
        self.first = np.squeeze(normalize(initial_state + noise()))
        self.second = np.squeeze(normalize(initial_state + noise()))
        self.collapsed = False

    def _update_qubit(self, value, second=False):
        if second:
            self.second = np.squeeze(np.asarray(value))

        else:
            self.first = np.squeeze(np.asarray(value))

    def measure(self, basis, second=False):
        qubit = self.second if second else self.first
        measured_bit = basis.measure(qubit)
        self._update_qubit(basis.basis_vectors[measured_bit], second)

        if not self.collapsed:
            self.collapsed = True
            entangled_value = basis.basis_vectors[1 - measured_bit]
            self._update_qubit(entangled_value, not second)

        return measured_bit


class E91Agent(object):
    def __init__(self, bases, second=False):
        self.bases = bases
        self.second = second
        self.used_bases = []
        self.bits = []

    def measure_qubits(self, qubits):
        for qubit in qubits:
            self.measure(qubit)

    def measure(self, qubit):
        basis = np.random.choice(self.bases)
        self.used_bases.append(basis)
        self.bits.append(qubit.measure(basis, self.second))

    def get_test_bases_and_bits(self, other_used_bases):
        test_indices = np.where(np.array(self.used_bases) !=
                                np.array(other_used_bases))[0]
        return test_indices, np.take(self.used_bases, test_indices),\
               np.take(self.bits, test_indices)


def correlation_test(alice, bob):
    test_indices, alice_bases, alice_bits = \
        alice.get_test_bases_and_bits(bob.used_bases)
    _, bob_bases, bob_bits = bob.get_test_bases_and_bits(alice.used_bases)

    results = defaultdict(lambda: defaultdict(int))
    test_statistic = 0

    for i in range(len(alice_bases)):
        a_base = alice_bases[i]
        b_base = bob_bases[i]

        if a_base == alice.bases[1] or b_base == bob.bases[0]:
            continue

        a_bit = alice_bits[i]
        b_bit = bob_bits[i]

        results[a_base, b_base][a_bit, b_bit] += 1

    for a_base in alice.bases:
        for b_base in bob.bases:
            if a_base == alice.bases[1] or b_base == bob.bases[0]:
                continue

            current = results[a_base, b_base]
            current_total = sum(current.values())
            correlation = float(current[0, 0] + current[1, 1] -
                           current[0, 1] - current[1, 0]) / current_total

            if a_base == alice.bases[2] and b_base == bob.bases[2]:
                correlation *= -1

            test_statistic += correlation

    return test_statistic


def rot_basis(theta, basis_vectors):
    rotation = np.matrix([[np.cos(theta), -1 * np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])

    return (basis_vectors[0] * rotation), (basis_vectors[1] * rotation)


def generate_alice_bob(noise=ZERO_NOISE_LAMBDA):
    alice = E91Agent((QuantumBasis(COMPUTATIONAL_VECS, '0', noise),
                      QuantumBasis(rot_basis(np.pi / 8, COMPUTATIONAL_VECS),
                                   'PI / 8', noise),
                      QuantumBasis(rot_basis(np.pi / 4, COMPUTATIONAL_VECS),
                                   'PI / 4', noise)))
    bob = E91Agent((QuantumBasis(COMPUTATIONAL_VECS, '0', noise),
                    QuantumBasis(rot_basis(np.pi / 8, COMPUTATIONAL_VECS),
                                 'PI / 8', noise),
                    QuantumBasis(rot_basis(np.pi / -8, COMPUTATIONAL_VECS),
                                 '-PI / 8', noise)),
                   second=True)
    return alice, bob


def e91(n=DEFAULT_NUM_BITS, eve_before_alice=False,
        eve_before_bob=False, noise=ZERO_NOISE_LAMBDA):
    alice, bob = generate_alice_bob(noise)

    qubits = [QubitPair(noise=noise) for _ in range(n)]

    if eve_before_alice:
        eve_basis = QuantumBasis(COMPUTATIONAL_VECS, '0', noise=noise)
        [qubit.measure(eve_basis) for qubit in qubits]

    alice.measure_qubits(qubits)

    if eve_before_bob:
        eve_basis = QuantumBasis(COMPUTATIONAL_VECS, '0', noise=noise)
        [qubit.measure(eve_basis, second=True) for qubit in qubits]

    bob.measure_qubits(qubits)
    return correlation_test(alice, bob)


def generate_gaussian_noise(mu=DEFAULT_NOISE_MU,
                            sigma=DEFAULT_NOISE_SIGMA,
                            size=NOISE_SIZE):
    return partial(np.random.normal, mu, sigma, size)


if __name__ == '__main__':
    print e91()
    print e91(eve_before_alice=True)
    print e91(eve_before_bob=True)
    print e91(noise=generate_gaussian_noise(sigma=0.05))
    print e91(eve_before_alice=True, noise=generate_gaussian_noise(sigma=0.05))
