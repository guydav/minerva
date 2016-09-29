import heapq
import random

N = 1000  # Number of atoms
p_hit = 0.9  # Probability that a particle will hit another atom
n_particles = 2  # Number of particles released
halflife = 1e6  # halflife of the atoms:
collision_time = 1e-6  # Time for collision:


def decay(has_decayed, time, index, heap):
    if has_decayed[index]:
        return  # Already decayed, return early
    # now the atom has decayed
    print "Atom %d decayed at time: %3.3f" % (index, current_time)
    has_decayed[index] = True
    for n in range(n_particles):
        if random.random() < p_hit:
            # The particle hit another atom
            next_time = time + collision_time
            next_index = random.randint(0, N - 1)
            # Schedule the decay of atom next_index at time next_time
            heapq.heappush(heap, (next_time, next_index))


if __name__ == '__main__':
    has_decayed_list = [False for a in range(N)]
    times = [random.expovariate(1.0 / halflife) for a in range(N)]
    events = zip(times, range(N))
    # Create a heap containing the times of all the atoms decays:
    collision_heap = [(times[i], i) for i in xrange(len(times))]
    heapq.heapify(collision_heap)

    # Now implement a continuous time simulation using a heap:
    while collision_heap:
        (current_time, current_index) = heapq.heappop(collision_heap)
        decay(has_decayed_list, current_time, current_index, collision_heap)