import numpy as np
from scipy import stats

ONE_OVER_SQRT_TWO = 1.0 / (2 ** 0.5)
COMPUTATIONAL_NAME = 'Computational'
COMPUTATIONAL_VECTORS = (np.array((1, 0)),
                         np.array((0, 1)))
HADAMARD_NAME = 'Hadamard'
HADAMARD_VECTORS = (np.array((ONE_OVER_SQRT_TWO, ONE_OVER_SQRT_TWO)),
                    np.array((ONE_OVER_SQRT_TWO, -1 * ONE_OVER_SQRT_TWO)))
DEFAULT_NUM_BITS = 256
TEST_BITS_PROPORTION = 0.5
SIGNIFICANCE_LEVEL = 0.05
EVE_NAME = 'Eve'
EVE_VECTORS = (np.array((np.cos(np.pi / 8), np.sin(np.pi / 8))),
               np.array((-1 * np.sin(np.pi / 8), np.cos(np.pi / 8))))


class QuantumBasis(object):
    def __init__(self, basis_vectors, name):
        self.basis_vectors = basis_vectors
        self.name = name

    def generate(self, bit_value=None):
        if bit_value is None:
            bit_value = np.random.choice(2)

        return self.basis_vectors[bit_value]

    def interpret(self, bit_value):
        return self.basis_vectors[bit_value]

    def measure(self, qubit):
        projections = [abs(vec.dot(qubit)) for vec in self.basis_vectors]
        probabilities = [proj ** 2 for proj in projections]
        return int(np.random.choice(2, p=probabilities))

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


COMPUTATIONAL_BASIS = QuantumBasis(COMPUTATIONAL_VECTORS, COMPUTATIONAL_NAME)
HADAMARD_BASIS = QuantumBasis(HADAMARD_VECTORS, HADAMARD_NAME)
BASES = (COMPUTATIONAL_BASIS, HADAMARD_BASIS)
EVE_BASIS = QuantumBasis(EVE_VECTORS, EVE_NAME)


class BB84Agent(object):
    def __init__(self, bases=BASES):
        self.bases = bases
        self.used_bases = np.array(())
        self.bits = np.array(())
        self.matching_bases = None
        self.key_indices = None

    def send_qubits(self, n=1):
        bases = np.random.choice(self.bases, size=n)
        self.used_bases = np.append(self.used_bases, bases)

        bits = np.random.choice(2, size=n)
        self.bits = np.append(self.bits, bits)

        return [basis.generate(bit) for basis, bit in zip(bases, bits)]

    def measure(self, qubits):
        bases = np.random.choice(self.bases, size=len(qubits))
        self.used_bases = np.append(self.used_bases, bases)
        measured_bits = [basis.measure(qubit)
                         for basis, qubit
                         in zip(bases, qubits)]
        self.bits = np.append(self.bits, measured_bits)

    def compare_used_bases(self, other):
        self.matching_bases = np.where(self.used_bases == other.used_bases)[0]

    def pick_test_bits(self, prop=TEST_BITS_PROPORTION):
        test_bit_indices = np.random.choice(
            self.matching_bases,
            size=int(len(self.matching_bases) * prop),
            replace=False)

        self.key_indices = np.setdiff1d(self.matching_bases, test_bit_indices)

        return test_bit_indices, np.take(self.bits, test_bit_indices)

    def test_bits(self, locations, bits):
        self.key_indices = np.setdiff1d(self.matching_bases, locations)

        my_bits = np.take(self.bits, locations)
        n = float(len(bits))
        p_error = sum(bits != my_bits) / n
        std_dev = (p_error * (1 - p_error)) / n

        if 0 == std_dev:
            return 0, 0, 1

        t_score = p_error / std_dev
        return p_error, t_score, stats.t(df=n - 1).sf(t_score)


def simulate_bb84(n=DEFAULT_NUM_BITS, eve=False):
    alice = BB84Agent()
    bob = BB84Agent()
    # Alice picks n random bits and bases, and sends each bit in a base
    qubits = alice.send_qubits(n)

    if eve:
        qubits = [EVE_BASIS.generate(EVE_BASIS.measure(qubit))
                  for qubit in qubits]

    # Bob picks n random bases, and measures each bit in a basis
    bob.measure(qubits)

    # Bob and Alice exchange basis lists, and now have ~n/2 bits to play with
    alice.compare_used_bases(bob)
    bob.compare_used_bases(alice)

    # Alice sends Bob ~half of those and the locations, Bob checks that they
    # match, or runs some statistical test to ascertain a probability of
    # eavesdropping
    indices, bits = alice.pick_test_bits()
    print bob.test_bits(indices, bits)

    # If test is successful, the remaining bits (~n/4) are the OTP key

    # Information reconciliation

    # Privacy amplification


if __name__ == '__main__':
    simulate_bb84()
    simulate_bb84(eve=True)
