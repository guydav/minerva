CACHE_KEY_SEPARATOR = '|'
CACHE = {}

# -- Caching Wrapper -- #

def cache(ignore_order=False):
    """
    A decorator supporting caching of a function with any number of arguments

    I had this ready from a previous project where I also had to compute and cache distance

    :param ignore_order: Is the function argument-order sensitive, or not.
    :return: The same function, wrapped to cache
    """

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

    # This could be made generic, but for two drivers only it is easier to hard-code the paths
    options = [(a, b, c, d), (a, c, b, d), (a, c, d, b), (c, d, a, b), (c, a, d, b), (c, a, b, d)]

    shortest_distance = 


@cache(True)
def distance(start, finish):
    '''
    Easily generalizable to higher dimensions, if needed, but for clarity currently hard-coded to 2d
    :param start: The point to calculate distance from, an (x, y) tuple
    :param finish: The point to calculate distance to, an (x, y) tuple
    :return: The euclidean distance between them
    '''
    return ((start[0] - finish[0]) ** 2 + (start[1] - finish[1]) ** 2) ** 0.5