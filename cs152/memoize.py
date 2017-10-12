from collections import defaultdict
import numpy as np


# The global cache used by the caching decotrator
CACHE = defaultdict(dict)


def memoize_heuristic(func):
    """
    A decorator supporting caching of a heuristic function, accepting a state as an argument
    Assumes the state parameter is a numpy.array
    :return: The same function, wrapped to cache
    """
    global CACHE
    function_cache = CACHE[func]

    def cached_function(state):
        key = tuple(int(x) for x in np.nditer(state))

        if key in function_cache:
            return function_cache[key]

        result = func(state)
        function_cache[key] = result

        return result

    cached_function.__name__ = 'memoized({name})'.format(name=func.__name__)
    return cached_function


def minimal_memoize(func):
    """
    A decorator supporting caching of a function with any number of arguments
    :param ignore_order: Is the function argument-order sensitive, or not.
    :return: The same function, wrapped to cache
    """
    global CACHE
    function_cache = CACHE[func]

    def cached_function(*args):
        if args in function_cache:
            return function_cache[args]

        result = func(*args)
        function_cache[args] = result

        return result

    return cached_function


def memoize_group_swap(func):
    """
    A decorator written specifically to memoize the group swap function.
    Splits to separate sub-caches by the first argument (the target function),
    and then sorts each row / column and queries the cache for it. Sorting is 
    required since we do not care about order within the row, and want to maintain
    each combination (ignoring order) only once.
    """
    global CACHE
    function_cache = CACHE[func]

    def cached_group_swap(target_func, groups):
        if target_func not in function_cache:
            function_cache[target_func] = {}

        target_func_cache = function_cache[target_func]
        key = tuple([tuple(sorted(grp)) for grp in groups])

        if key in target_func_cache:
            return target_func_cache[key]

        result = func(target_func, np.array(groups))
        target_func_cache[key] = result

        return result

    cached_group_swap.__name__ = 'memoized({name})'.format(name=func.__name__)
    return cached_group_swap


def memoize_generic(ignore_order=False):
    """
    A decorator supporting caching of a function with any number of arguments
    :param ignore_order: Is the function argument-order sensitive, or not.
    :return: The same function, wrapped to cache
    """

    def cache_function(func):
        global CACHE
        function_cache = CACHE[func]

        def cached_function(*args, **kwargs):
            key = list(args) + list(kwargs.items())

            if ignore_order:
                key.sort()

            key = tuple(key)

            if key in function_cache:
                return function_cache[key]

            result = func(*args, **kwargs)
            function_cache[key] = result

            return result

        return cached_function

    return cache_function
