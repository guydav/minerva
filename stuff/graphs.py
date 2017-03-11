import networkx as nx
from collections import defaultdict
import heapq

WEIGHT = 'weight'
COST = 'cost'

NEWARK = 'Newark'
WOODBRIDGE = 'Woodbridge'
TRENTON = 'Trenton'
ASBURY_PARK = 'Asbury Park'
CAMDEN = 'Camden'
ATLANTIC_CITY = 'Atlantic City'
CAPE_MAY = 'Cape May'

NEW_JERSEY_DATA = {NEWARK: {WOODBRIDGE: {WEIGHT: 20, COST: 0.60}},
                 WOODBRIDGE: {TRENTON: {WEIGHT: 42, COST: 1.00},
                             ASBURY_PARK: {WEIGHT: 35, COST: 0.75},
                             CAMDEN: {WEIGHT: 60, COST: 0.00}},
                 TRENTON: {ASBURY_PARK: {WEIGHT: 40, COST: 0.00},
                          CAMDEN: {WEIGHT: 30, COST: 0.70}},
                 ASBURY_PARK: {ATLANTIC_CITY: {WEIGHT: 75, COST: 1.35}},
                 CAMDEN: {ATLANTIC_CITY: {WEIGHT: 55, COST: 1.25},
                         CAPE_MAY: {WEIGHT: 85, COST: 0.00}},
                 ATLANTIC_CITY: {CAPE_MAY: {WEIGHT: 45, COST: 0.75}}
                }

new_jersey_graph = nx.from_dict_of_dicts(NEW_JERSEY_DATA)


class MinHeap(object):
    def __init__(self, data=None):
        if data is not None:
            self.heap = heapq.heapify(data)

        else:
            self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    def push_pop(self, item):
        return heapq.heappushpop(self.heap, item)

    def n_smallest(self, n):
        return heapq.nsmallest(n, self.heap)

    def n_largest(self, n):
        return heapq.nsmallest(n, self.heap)

    def peek(self):
        return self.heap[0]

    def heapify(self):
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str(self.heap)

    def __getitem__(self, item):
        return self.heap.__getitem__(item)


def dijkstra(graph, start, finish, cost_function):
    frontier = MinHeap()
    frontier.push((start, 0))
    came_from = {start: (None, 0)}

    while len(frontier):
        current, current_cost = frontier.pop()

        if current == finish:
            print came_from
            path = [current]
            while current != start:
                current = came_from[current][0]
                path.append(current)
            path.reverse()
            return path

        neighbors = graph.neighbors(current)

        for neighbor in neighbors:
            neighbor_cost = current_cost + cost_function(
                graph.get_edge_data(current, neighbor))

            # Visited before, and new cost higher than old cost:
            if neighbor in came_from and neighbor_cost >= came_from[neighbor][1]:
                continue

            came_from[neighbor] = (current, neighbor_cost)
            frontier.push((neighbor, neighbor_cost))

    return None


def distance_cost_function(data_dict):
    return data_dict[WEIGHT]


def price_cost_function(data_dict):
    return data_dict[COST]


if __name__ == '__main__':
    print dijkstra(new_jersey_graph, NEWARK, CAPE_MAY, price_cost_function)
