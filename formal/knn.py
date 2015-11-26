import numpy as np
import progressbar
import sortedcontainers
import random
import matplotlib.pyplot as plot

# -- Constants -- #

DATA_PATH = 'knn_data.npy'

CACHE_KEY_SEPARATOR = '|'
CACHE = {}

# SORTED_DATA_CACHE = []
DISTANCE_CACHE = {}

POTENTIAL_NEAREST_POINTS_HEURISTIC = 2

LABELS = {}


# -- Caching Wrapper -- #

def cache(should_ignore_order):
    """
    A decorator supporting caching of a function with any number of arguments
    :param function: The function whose outputs we want to cache
    :return: The same function, wrapped to cache
    """

    def cache_function(function):
        global CACHE

        if function not in CACHE:
            CACHE[function] = {}

        function_cache = CACHE[function]

        def cached_function(*args, **kwargs):
            key = ''
            # Doing some tricks here to ignore ordering
            if args:
                if should_ignore_order:
                    key += CACHE_KEY_SEPARATOR.join([str(arg) for arg in sorted(args)])

                else:
                    key += CACHE_KEY_SEPARATOR.join([str(arg) for arg in args])

            # Doing some tricks here to ignore ordering
            if kwargs:
                if should_ignore_order:
                    key += CACHE_KEY_SEPARATOR.join(
                        ['{name}:{value}'.format(name=name, value=kwargs[name]) for name in sorted(kwargs.keys())])

                else:
                    key += CACHE_KEY_SEPARATOR.join(
                        ['{name}:{value}'.format(name=name, value=kwargs[name]) for name in kwargs.keys()])


            if key in function_cache:
                return function_cache[key]

            result = function(*args, **kwargs)
            if key:
                function_cache[key] = result

            return result

        return cached_function

    return cache_function

# def cache(function):
#     """
#     A decorator supporting caching of a function with any number of arguments
#     :param function: The function whose outputs we want to cache
#     :return: The same function, wrapped to cache
#     """
#     global CACHE
#
#     if function not in CACHE:
#         CACHE[function] = {}
#
#     function_cache = CACHE[function]
#
#     def cached_function(*args, **kwargs):
#         key = ''
#         # Doing some tricks here to ignore ordering
#         if args:
#             key += CACHE_KEY_SEPARATOR.join([str(arg) for arg in args])
#
#         # Doing some tricks here to ignore ordering
#         if kwargs:
#             key += CACHE_KEY_SEPARATOR.join(
#                 ['{name}:{value}'.format(name=name, value=kwargs[name]) for name in sorted(kwargs.keys())])
#
#         if key in function_cache:
#             return function_cache[key]
#
#         result = function(*args, **kwargs)
#         if key:
#             function_cache[key] = result
#
#         return result
#
#     return cached_function


# -- Classes -- #

class PriorityQueue(object):
    def __init__(self):
        self.priority_to_elements = sortedcontainers.SortedDict()
        self.element_to_priority = sortedcontainers.SortedDict()


    def put(self, element, priority):
        '''
        Puts an element into the priority queue
        :param element: The element to insert. Can be anything.
        :param priority: The priority, as a number. Lowered is assumed to be better / higher priority.
        :return:
        '''
        if (element in self.element_to_priority) and (priority > self.element_to_priority[element]):
            return

        self.element_to_priority[element] = priority

        if priority not in self.priority_to_elements:
            self.priority_to_elements[priority] = []

        self.priority_to_elements[priority].append(element)


    def get_extreme_key(self, min=True):
        keys = self.priority_to_elements.keys()

        if min:
            key = keys[0]

        else:
            key = keys[len(keys) - 1]

        return key


    def get_extreme_priority(self, min=True):
        key = self.get_extreme_key(min)
        return self.priority_to_elements[key]


    def get(self, min=True):
        '''
        Removes and returns the element with the lowest priority score (by default).
        :param min: True if should return the object with the lowest priority score. False if should return the highest.
        :return: The most extremely prioritized object, base on whether the parameter min is True or False
        '''
        key = self.get_extreme_key(min)
        priority =  self.priority_to_elements[key]

        element = priority.pop()

        if not priority:
            del self.priority_to_elements[key]

        return element


    def get_many(self, count, min=True):
        return [self.get(min) for i in xrange(count)]


    def __len__(self):
        return len(self.priority_to_elements)



class Point(object):
    def __init__(self, line):
        '''
        Assumes each line has n coordinates up until the last item, which is the label
        :param line: an iterable/slicable containing the values for a given datum
        :return: the Point objecy
        '''
        if len(line) < 2:
            raise ValueError('Each line must have at least one coordinate and an actual label')

        self.coordinates = line[:len(line) - 1]
        self.actual_label = line[len(line) - 1]
        self.predicted_labels = {}

        global LABELS

        if not self.actual_label in LABELS:
            LABELS[self.actual_label] = 0

        LABELS[self.actual_label] += 1


    def __repr__(self):
        return '({coordinates}) => {label}'.format(coordinates=', '.join([str(x) for x in self.coordinates]),
                                                   label=self.actual_label)


# -- Prediction Functions -- #

def read_data(data_path=DATA_PATH):
    '''
    Read the data from the an input fle, and convert it to Point objects
    :param data_path:
    :return:
    '''
    with open(data_path) as data_file:
        data = np.load(data_file)

    return [Point(line) for line in data]


def build_distance_cache(data):
    '''

    :param data:
    :return:
    '''
    global DISTANCE_CACHE

    print 'Building point-distance cache'

    progress = progressbar.ProgressBar()

    for point in progress(data):
        data_for_point = data[:]
        data_for_point.remove(point)
        data_for_point.sort(key=lambda p: distance(p, point))
        DISTANCE_CACHE[point] = data_for_point

# def build_sorted_data_cache(data):
#     '''
#
#     :param data:
#     :return:
#     '''
#     global SORTED_DATA_CACHE
#
#     for dimension in xrange(len(data[0].coordinates)):
#         SORTED_DATA_CACHE.append(sortedcontainers.SortedList(data, key=lambda p: p.coordinates[dimension]))


# def get_potential_nearest_points(k, point):
#     '''
#
#     :param k:
#     :param point:
#     :return:
#     '''
#     global SORTED_DATA_CACHE
#     potential_nearest_points = set()
#
#     extra_items = 0
#     while len(potential_nearest_points) < POTENTIAL_NEAREST_POINTS_HEURISTIC * k:
#         for dimension in xrange(len(point.coordinates)):
#             dimension_list = SORTED_DATA_CACHE[dimension]
#             bisect_index = dimension_list.bisect(point)
#             potential_nearest_points.update(dimension_list[max(bisect_index - k - extra_items, 0) :
#                 min(bisect_index + k + extra_items, len(dimension_list))])
#
#         extra_items +=POTENTIAL_NEAREST_POINTS_HEURISTIC
#
#     if point in potential_nearest_points:
#         potential_nearest_points.remove(point)
#
#     return potential_nearest_points


@cache(True)
def distance(first_point, second_point):
    return pythag_distance(first_point, second_point)


def pythag_distance(first_point, second_point):
    return sum(map(lambda pair: (pair[0] - pair[1]) ** 2,
                   zip(first_point.coordinates, second_point.coordinates))) ** 0.5


def grid_distance(first_point, second_point):
    return sum(map(lambda pair: (pair[0] - pair[1]),
                   zip(first_point.coordinates, second_point.coordinates)))


def find_nearest_neighbors(k, data, point):
    '''
    Find the nearest neighbors - using a sorted cache
    :param k:
    :param data:
    :return:
    '''
    global DISTANCE_CACHE

    if not DISTANCE_CACHE:
        build_distance_cache(data)

    return DISTANCE_CACHE[point][:k]

# def find_nearest_neighbors(k, data, point):
#     '''
#     Find the nearest neighbors - using a sorted cache
#     :param k:
#     :param data:
#     :return:
#     '''
#     global SORTED_DATA_CACHE
#
#     if not SORTED_DATA_CACHE:
#         build_sorted_data_cache(data)
#
#     potential_nearest_points = get_potential_nearest_points(k, point)
#     distance_to_points = PriorityQueue()
#
#     for potential_nearest in potential_nearest_points:
#         distance_to_points.put(potential_nearest, distance(point, potential_nearest))
#
#     heuristic_nearest = distance_to_points.get_many(k)
#
#     all_data_dtp = PriorityQueue()
#     for p in data:
#         if p != point:
#             all_data_dtp.put(p, distance(point, p))
#
#     all_nearest = all_data_dtp.get_many(k)
#
#     print sum([distance(point, d) for d in heuristic_nearest]) - sum([distance(point, d) for d in all_nearest])
#
#     return all_nearest


def label_point(k, k_nearest, point, bias_lowest=True):
    '''

    :param k:
    :param k_nearest:
    :param point:
    :param bias_lowest:
    :return:
    '''
    global LABELS

    potential_labels = PriorityQueue()

    for label in LABELS:
        potential_labels.put(label, len([p for p in k_nearest if p.actual_label == label]))

    top_labels = potential_labels.get_extreme_priority(False)

    if len(top_labels) == 1:
        point.predicted_labels[k] = top_labels[0]

    else:
        top_labels.sort(key=lambda l: LABELS[l], reverse=bias_lowest)
        point.predicted_labels[k] = top_labels[0]


def predict_labels(k, data):
    '''
    Go through all data points, and predict the label given a specific k
    :param k: the k to currently consider
    :param data: the data (some iterable data structure including points)
    :return:
    '''
    if not (0 < k < len(data)):
        raise ValueError('Invalid k specified. k | 0 < k < len(data)')

    for point in data:
        k_nearest = find_nearest_neighbors(k, data, point)
        label_point(k, k_nearest, point, False)


def calculate_accuracy(k, data):
    correct = [p for p in data if p.predicted_labels[k] == p.actual_label]
    return float(len(correct)) / len(data) * 100


def predict_and_print_accuracy(min_k, max_k, data):
    top_k = 0
    top_accuracy = 0

    print 'Predicting for k in [{min},{max}]'.format(min=min_k, max=max_k)
    progress = progressbar.ProgressBar()

    for k in progress(xrange(min_k, max_k+1)):
        predict_labels(k, data)
        accuracy = calculate_accuracy(k, data)
        # print 'For k = {k}, the prediction accuracy is {accuracy:2.2f}%'.format(k=k, accuracy=accuracy)

        if accuracy > top_accuracy:
            top_accuracy = accuracy
            top_k = k

    print 'The best k found was {k}, its accuracy is {accuracy:2.2f}%'.format(k=top_k, accuracy=top_accuracy)
    return top_k


# -- Graphing Functions -- #

DEFAULT_HORIZONTAL_SIZE = 12
DEFAULT_VERTICAL_SIZE = 8
DEFAULT_DPI = 400
DEFAULT_MARKER_SIZE = 4
DEFAULT_ALPHA = 0.75

MARKER_OPTIONS = ['o', '*', 'x', 's', 'D', '|', '2', '^', '8', 'p']
SUCCESS_COLOR = 'green'
FAILURE_COLOR = 'red'
COLOR_OPTIONS = {True: SUCCESS_COLOR, False:FAILURE_COLOR}


def plot_single_series(axes, data_list, label, marker, is_correct=True):
    axes.scatter([p.coordinates[0] for p in data_list[label]], [p.coordinates[1] for p in data_list[label]],
        label='Label {label} predicted {status}'.format(label=str(int(label)), status=(is_correct and 'correctly' or 'incorrectly')),
        color=COLOR_OPTIONS[is_correct], marker=marker, alpha=DEFAULT_ALPHA)


def graph_prediction_results(k, data):
    if len(data[0].coordinates) != 2:
        raise ValueError('Graphing is currently only supported for 2d data...')

    global LABELS

    accurate_data = {}
    missed_data = {}

    for label in LABELS:
        accurate_data[label] = []
        missed_data[label] = []

    for p in data:
        label = p.actual_label

        if label == p.predicted_labels[k]:
            accurate_data[label].append(p)

        else:
            missed_data[label].append(p)

    figure, axes = plot.subplots()
    figure.set_size_inches(DEFAULT_HORIZONTAL_SIZE, DEFAULT_VERTICAL_SIZE)
    figure.set_dpi(DEFAULT_DPI)

    markers = MARKER_OPTIONS[:]
    random.shuffle(markers)

    for label in LABELS:
        marker = markers.pop()
        plot_single_series(axes, accurate_data, label, marker, True)
        plot_single_series(axes, missed_data, label, marker, False)

    axes.set_xlabel('X coordinate')
    axes.set_ylabel('Y coordinate')
    axes.legend(loc='upper left')
    axes.grid(True)

    plot.show()


def main():
    # global data
    # data = read_data()
    # best_k = predict_and_print_accuracy(1, 2 * int(len(data)**0.5), data)
    # graph_prediction_results(best_k, data)

    with open(DATA_PATH) as data_file:
        data = np.load(data_file)
        print dir(data[0])


if __name__ == '__main__':
    main()
