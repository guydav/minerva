import heapq
import random


class MedianHeap(object):
    def __init__(self):
        self.lower_half = []
        self.upper_half = []

    def _push_lower_half(self, item):
        heapq.heappush(self.lower_half, -1 * item)

    def _push_upper_half(self, item):
        heapq.heappush(self.upper_half, item)

    def _peek_lower_half(self):
        return -1 * self.lower_half[0]

    def _peek_upper_half(self):
        return self.upper_half[0]

    def _pop_lower_half(self):
        return -1 * heapq.heappop(self.lower_half)

    def _pop_upper_half(self):
        return heapq.heappop(self.upper_half)

    def __len__(self):
        return len(self.lower_half) + len(self.upper_half)

    def __nonzero__(self):
        return len(self) != 0

    def push(self, item):
        # Both are empty - push to one arbitrarily
        if not self.lower_half and not self.upper_half:
            self._push_lower_half(item)

        # Only upper is empty - compare to peek on lower half
        elif not self.upper_half:
            if item <= self._peek_lower_half():
                self._push_lower_half(item)
            else:
                self._push_upper_half(item)

        # Only lower is empty - compare to peek on upper half
        elif not self.lower_half:
            if item > self._peek_upper_half():
                self._push_upper_half(item)
            else:
                self._push_lower_half(item)

        # Both halves have content - compare both
        else:
            if item <= self._peek_lower_half():
                self._push_lower_half(item)
            else:
                self._push_upper_half(item)

    def median(self):
        upper_half_len = len(self.upper_half)
        lower_half_len = len(self.lower_half)

        # If they're equally sized, arbitrarily return the lower of the two
        if upper_half_len == lower_half_len:
            return self._pop_lower_half()

        elif upper_half_len > lower_half_len:
            for _ in xrange((upper_half_len - lower_half_len) / 2):
                self._push_lower_half(self._pop_upper_half())

            return self._pop_upper_half()

        else:
            for _ in xrange((lower_half_len - upper_half_len) / 2):
                self._push_upper_half(self._pop_lower_half())

            return self._pop_lower_half()

    def __repr__(self):
        return 'Lower: {lower}, Upper:{upper}'.format(lower=[-1 * x for x in self.lower_half], upper=self.upper_half)


class QuantileHeap(MedianHeap):
    def quantile(self, tau):
        if tau < 0 or tau > 100:
            raise ValueError('Quantile must bet 0 <= q <= 100, received {q}'.format(q=quantile))

        # Start with the edge cases - 0 and 100
        # In this case, we want everything in the upper half
        if tau == 0:
            while self.lower_half:
                self._push_upper_half(self._pop_lower_half())

            return self._pop_upper_half()

        # In this case, we want everything in the lower half
        if tau == 100:
            while self.upper_half:
                self._push_lower_half(self._pop_upper_half())

            return self._pop_lower_half()

        # In every other case, we want to balance the two heaps
        ratio = float(tau) / float(100 - tau)

        while (ratio * len(self.upper_half)) > len(self.lower_half):
            self._push_lower_half(self._pop_upper_half())

        while (ratio * len(self.upper_half)) <= len(self.lower_half):
            self._push_upper_half(self._pop_lower_half())

        return self._pop_upper_half()


def main():
    # mh = MedianHeap()
    # numbers = range(10)
    # random.shuffle(numbers)
    #
    # for num in numbers:
    #     mh.push(num)
    #     print num, mh
    #
    # while mh:
    #     print mh.median(), mh

    qh = QuantileHeap()
    numbers = range(100)
    random.shuffle(numbers)

    for num in numbers:
        qh.push(num)

    print qh.quantile(75)
    print qh.quantile(33)


if __name__ == '__main__':
    main()