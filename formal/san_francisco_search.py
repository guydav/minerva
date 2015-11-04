import os
import pickle
import collections
import sortedcontainers


# -- Constants -- #

EDGES_FILE_NAME = 'san_francisco_edges.pickle'
EDGES_FILE_PATH = os.path.join(os.path.dirname(__file__), EDGES_FILE_NAME)
START_VALUE = (1501, 4118)
FINISH_VALUE = (6173, 7065)


# -- Classes -- #


class Node(object):
    def __init__(self, value):
        self.value = value
        self.adjacent_nodes = []

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return '{value} => [{adjacent}]'.format(value=self.value,
                                                  adjacent=', '.join([str(node.value) for node in self.adjacent_nodes]))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False

        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class PriorityQueue(object):
    def __init__(self):
        self.priority_to_elements = sortedcontainers.SortedDict()
        self.element_to_priority = sortedcontainers.SortedDict()

    def put(self, element, priority):
        # Assuming a lower priority is better
        if (element in self.element_to_priority) and (priority > self.element_to_priority[element]):
            return

        self.element_to_priority[element] = priority

        if priority not in self.priority_to_elements:
            self.priority_to_elements[priority] = []

        self.priority_to_elements[priority].append(element)

    def get(self):
        min_key = self.priority_to_elements.keys()[0]
        min_priority = self.priority_to_elements[min_key]

        element = min_priority.pop()

        if not min_priority:
            del self.priority_to_elements[min_key]

        return element

    def __len__(self):
        return len(self.priority_to_elements)


# -- Helper Methods -- #


def get_all_nodes():
    with open(EDGES_FILE_PATH, 'rb') as f:
        edges = pickle.load(f)

    node_set = set([edge[0] for edge in edges]).union(set([edge[1] for edge in edges]))

    adjacency_map = {}
    for edge in edges:
        for node in edge:
            if node not in adjacency_map:
                adjacency_map[node] = []

        adjacency_map[edge[0]].append(edge[1])
        adjacency_map[edge[1]].append(edge[0])

    node_map = {n: Node(n) for n in node_set}
    for node, adjs in adjacency_map.items():
        node_map[node].adjacent_nodes = [node_map[n] for n in adjs]

    return node_map


# def write_out_edges():
#     with open(EDGES_FILE, 'wb') as f:
#         pickle.dump(get_all_nodes(), f)

def breadth_first_search(node_map, start, finish, breadth=True):
    frontier = collections.deque([start])
    # visited = set([start])
    # Using queued as I assume lookup in the frontier is inefficient
    # queued = set([start])
    came_from = {start: None}

    loop_counter = 0

    while len(frontier):
        loop_counter += 1

        # BFS
        if breadth:
            current = frontier.popleft()
        # DFS
        else:
            current = frontier.pop()

        # visited.add(current)

        if current == finish:
            return True, loop_counter, came_from

        # new_neighbors = filter(lambda n: (n not in visited) and (n not in queued) , current.adjacent_nodes)
        new_neighbors = filter(lambda n: (n not in came_from) , current.adjacent_nodes)
        for next_node in new_neighbors:
            came_from[next_node] = current
            frontier.append(next_node)

    return False, loop_counter, None

def depth_first_search(node_map, start_value, finish_value):
    return breadth_first_search(node_map, start_value, finish_value, False)

def a_star_heuristic(current, finish):
    return abs(current.value[0] - finish.value[0]) + abs(current.value[1] - finish.value[1])
    # return ((current.value[0] - finish.value[0]) ** 2 + (current.value[1] - finish.value[1]) ** 2) ** 0.5

def a_star_search(node_map, start, finish, breadth=True):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}

    loop_counter = 0

    while len(frontier):
        loop_counter += 1

        current = frontier.get()

        if current == finish:
            return True, loop_counter, came_from

        new_neighbors = filter(lambda n: (n not in came_from) , current.adjacent_nodes)

        for next_node in new_neighbors:
            if next_node not in came_from:
                came_from[next_node] = current
                frontier.put(next_node, a_star_heuristic(next_node, finish))

    return False, loop_counter, None

def find_path(came_from, finish):
    path = []
    current = finish

    while came_from[current]:
        previous = came_from[current]
        path.append(previous)
        current = previous

    path.reverse()
    return path

def print_path(path):
    return ' => '.join([str(node) for node in path])

def search_and_print_result(node_map, start_value, finish_value, search_func):
    if start_value not in node_map:
        raise ValueError('Start node {value} not found'.format(value=start_value))

    elif finish_value not in node_map:
        raise ValueError('Finish node {value} not found'.format(value=finish_value))

    start = node_map[start_value]
    finish = node_map[finish_value]

    success, iterations, came_from = search_func(node_map, start, finish)
    if success:
        path = find_path(came_from, node_map[finish_value])
        print '{func} found a path of length {length} after {iterations} iterations:'.format(
            func=search_func.func_name, length=len(path), iterations=iterations)
        print print_path(path)

    else:
        print '{func} failed to find a path, terminating after {iterations} iterations'.format(
            func=search_func.func_name, iterations=iterations)

    print

# -- main -- #

def main():
    node_map = get_all_nodes()
    search_and_print_result(node_map, START_VALUE, FINISH_VALUE, breadth_first_search)
    search_and_print_result(node_map, START_VALUE, FINISH_VALUE, depth_first_search)
    search_and_print_result(node_map, START_VALUE, FINISH_VALUE, a_star_search)

if __name__ == '__main__':
    main()