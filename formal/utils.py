import sortedcontainers

CACHE_KEY_SEPARATOR = '|'
CACHE = {}



class PriorityQueue(object):
    def __init__(self):
        '''
        A simple priority queue implementation, using two sorted dictionaries.
        :return:
        '''
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


def cache(ignore_order=False):
    """
    A decorator supporting caching of a function with any number of arguments
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