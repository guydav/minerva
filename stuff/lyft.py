"""
Note - i used TODO's to mark places where I would improve upon this solution for more generality or better behavior
"""
CACHE_KEY_SEPARATOR = '|'
CACHE = {}

INVALID_ARGUMENTS = -1

# -- Caching Wrapper -- #


def cache(ignore_order=False):
    """
    A decorator supporting caching of a function with any number of arguments, sensitive or insensitive to order
    For functions such as a euclidean distance, the order insensitive caching is fantastic.

    I had this lying around from a previous project where I also had to compute and cache distances.

    :param ignore_order: Is the function argument-order sensitive, or not.
    :return: The same function, wrapped to cache
    """
    # TODO: This could probably use some escaping of the cache key separator
    # TODO: Alternatively, I could convert the kwargs to a tuple, instead of stringifying everything
    def cache_function(function):
        global CACHE

        if function not in CACHE:
            CACHE[function] = {}

        function_cache = CACHE[function]

        def cached_function(*args, **kwargs):
            key = ''
            # sorting to ignore ordering, using strings to easily support the kwargs as well
            if args:
                if ignore_order:
                    key += CACHE_KEY_SEPARATOR.join([str(arg) for arg in sorted(args)])

                else:
                    key += CACHE_KEY_SEPARATOR.join([str(arg) for arg in args])

            if kwargs:
                if ignore_order:
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


# -- Distance Function -- #


def shortest_detour(a, b, c, d):
    """
    Calculate the minimal distance that one of two drivers, one driving from A to B, another from C to D, would have
    to take to pick-up and drop-off the other drivers

    This assumes the second passenger can be picked up before the first is dropped off

    This could be easily adapted to other measures of distance (such as city-grid) by changing the distance function

    :param a: The starting point of the first driver, an (x, y) tuple
    :param b: The destination of the second driver, an (x, y) tuple
    :param c: The starting point of the second driver, an (x, y) tuple
    :param d: The destination of the second driver, an (x, y) tuple
    :return: The minimal distance and the path required to take it
    """
    if a == b or c == d:
        # Asking about a passenger being picked up and dropped off at the same place makes no sense
        return INVALID_ARGUMENTS, None


    # TODO: If I were going to make this more generic, I would generate the options dynamically rather than hardcoding
    # TODO: Probably be changing the signature of this method to accept (start, end) tuples, and then iterating over all
    # TODO: combinations thereof that preserve order within each tuple

    options = [(a, b, c, d), (a, c, b, d), (a, c, d, b), (c, d, a, b), (c, a, d, b), (c, a, b, d)]
    distances = [total_distance(option) for option in options]

    min_distance = min(distances)
    if distances.count(min_distance) == 1:
        return min_distance, options[distances.index(min_distance)]

    else:
        min_options = [options[i] for i in xrange(len(options)) if distances[i] == min_distance]
        return min_distance, tie_resolution_heuristic(a, b, c, d, min_options)


def tie_resolution_heuristic(a, b, c, d, min_options):
    """
    If we have a tie, it should preferably be resolved intelligently. This heuristic prefers route in which one
    passenger is picked up and dropped off before the other is picked up and dropped off. The main benefit is that
    the driver-pairing can be performed again, if another match is found en route to the second passenger's destination.

    Socially, it may prove a hit or miss, depending if passengers tend to prefer to ride alone or with strangers.

    If both options match the heuristic, the first one is chosen

    :param a: The starting point of the first driver, an (x, y) tuple
    :param b: The destination of the second driver, an (x, y) tuple
    :param c: The starting point of the second driver, an (x, y) tuple
    :param d: The destination of the second driver, an (x, y) tuple
    :param min_options: The options with the minimal distance
    :return: The chosen option
    """
    # TODO: If I make the option generation generic, I would similarly make this generic, generating only these options
    if (a, b, c, d) in min_options:
        return a, b, c, d

    if (c, d, a, b) in min_options:
        return c, d, a, b

    return min_options[0]


def total_distance(option):
    """
    Calculate the total distance for an option. Refactored out of shortest_detour aside as a function for clarity
    :param option: An iterable containing the the different points to pass through in an option
    :return: The total distance, starting at the first and finishing at the last
    """
    return sum([distance(option[i - 1], option[i]) for i in xrange(1, len(option))])


@cache(True)
def distance(start, finish):
    """
    Calculate the euclidean distance between two points
    :param start: The point to calculate distance from, an (x, y) tuple
    :param finish: The point to calculate distance to, an (x, y) tuple
    :return: The euclidean distance between them
    """
    # TODO: Optimally, this would be a better measure of distance, such as a Google Maps estimate
    return ((start[0] - finish[0]) ** 2 + (start[1] - finish[1]) ** 2) ** 0.5


if __name__ == '__main__':
    print shortest_detour((0,0), (1,1), (2,2), (3,3))