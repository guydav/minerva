__author__ = 'guydavidson'

# -- Imports -- #

IMPORT_ERROR_MESSAGE = 'Failed to import {library}, proceeding without it'

try:
    import tabulate
except ImportError, e:
    tabulate = None
    print IMPORT_ERROR_MESSAGE.format(library='tabulate')

try:
    import numpy
except ImportError, e:
    numpy = None
    print IMPORT_ERROR_MESSAGE.format(library='numpy')

try:
    import scipy.stats
except ImportError, e:
    scipy = None
    print IMPORT_ERROR_MESSAGE.format(library='scipy')

try:
    import matplotlib.pyplot as plot
except ImportError, e:
    plot = None
    print IMPORT_ERROR_MESSAGE.format(library='matploblib')

try:
    import sklearn.cluster
except ImportError, e:
    sklearn = None
    print IMPORT_ERROR_MESSAGE.format(library='sklearn')

# -- Constants -- #

DEFAULT_DATA = [('annual average temperature anomaly', 'annual precipitation anomaly', 'Palmer Drought Index anomaly',
                 'cooling degree days anomaly', 'heating degree days anomaly'), (-0.41, -0.24, 0.34, 22, 115),
                (-0.36, 0.39, 0.2, -18, 2), (-0.65, 1.76, 1.58, -57, 136), (0.27, 5.02, 4.45, 25, -223),
                (0.24, 0.68, 3.07, -81, -154), (-0.52, 3.09, 3.32, -34, -86), (-0.55, -3.63, -0.75, -173, 148),
                (0.53, 0.5, -1.55, 88, 14), (-0.97, 0.2, 0.52, 31, 360), (-1.14, 2.81, 1.83, -87, 192),
                (0.37, -1.67, -0.8, 110, 101), (1.1, 0.02, -1.16, 9, -92), (-0.67, 3.92, 2.97, -58, 17),
                (-0.14, 4.82, 4.61, 48, 19), (-0.04, 1.46, 3.55, 17, -105), (-0.72, 0.03, 0.56, 7, 23),
                (1.3, 1.44, 1.74, 59, -344), (1.31, -0.93, -0.97, 72, -296), (0.61, -4.04, -3.91, 91, -8),
                (-0.18, -0.89, -1.25, -27, 47), (1.49, 2.23, -0.37, 76, -619), (1.14, 2.5, 0.75, 140, -445),
                (0.58, 1.32, 0.79, -135, -228), (-0.76, 2.68, 3.97, 23, 38), (0.85, 0.68, 1.74, 38, -191),
                (0.63, 2.75, 2.4, 103, -150), (-0.13, 3.76, 2.39, 8, 13), (0.18, 1.92, 3.9, -25, -138),
                (2.21, 3.95, 1.15, 236, -684), (1.86, -1.47, -0.54, 117, -516), (1.25, -1.72, -4.84, 70, -247),
                (1.68, -0.92, -3.27, 86, -470), (1.19, -0.89, -2.46, 190, -360), (1.24, 0.57, 0.18, 91, -175),
                (1.08, 3.31, 0.91, 32, -361), (1.62, 0.14, -0.31, 199, -349), (2.23, -0.12, -2.62, 172, -648),
                (1.63, -0.76, -2.05, 199, -412), (0.27, 1.3, -0.05, 85, -181), (0.37, 2.36, 0.85, 44, -189),
                (0.96, 1.43, 2.65, 256, -198), (1.16, 0.16, -0.56, 268, -348), (3.26, -2.41, -4.41, 291, -891),
                (0.41, 1.12, -0.62, 100, -178), (0.51, 0.9, 0.84, 85, -80)]

DEFAULT_TEST_THRESHOLD = 10 ** -10
CACHE_KEY_SEPARATOR = '|'

NAME_FIELD = 'Name'
MEAN_FIELD = 'Mean'
MEDIAN_FIELD = 'Median'
MODE_FIELD = 'Mode'
VARIANCE_FIELD = 'Variance'
STD_DEV_FIELD = 'Std Dev'

CACHE = {}


# -- Caching Wrapper -- #

def cache(function):
    """
    A decorator supporting caching of a function with any number of arguments
    :param function: The function whose outputs we want to cache
    :return: The same function, wrapped to cache
    """
    if function not in CACHE:
        CACHE[function] = {}

    function_cache = CACHE[function]

    def cached_function(*args, **kwargs):
        key = ''
        if args:
            key += CACHE_KEY_SEPARATOR.join([str(arg) for arg in args])

        # Doing some tricks here to ignore ordering
        if kwargs:
            key += CACHE_KEY_SEPARATOR.join(
                ['{name}:{value}'.format(name=name, value=kwargs[name]) for name in sorted(kwargs.keys())])

        if key in function_cache:
            return function_cache[key]

        result = function(*args, **kwargs)
        if key:
            function_cache[key] = result

        return result

    return cached_function


# -- Stats Functions -- #

@cache
def mean(data):
    """
    Calculates the mean of an iterable of numerical data
    :param data: An iterable containing numeric data
    :return: The mean of the data, or None if received an empty list
    """
    if not data:
        return None

    return float(sum(data)) / len(data)


@cache
def median(data):
    """
    Calculate the median of the data, without reordering the original list.
    We define the median of an even-sized list to be the mean of the middle two elements
    :param data: An iterable containing numeric data
    :return: The median of the data, or None if received an empty list
    """
    if not data:
        return None

    sorted_data = sorted(data)
    length = len(sorted_data)

    # Odd length:
    if length % 2:
        return sorted_data[length / 2]

    # Even:
    else:
        middle_start = length / 2 - 1
        return mean(sorted_data[middle_start:middle_start + 2])


@cache
def mode(data):
    """
    Calculate the mode of the data.
    We define the mode as the element (or set of elements) appearing most often in the data.
    :param data: An iterable containing numeric data
    :return: The median of the data, or None if received an empty list
    """
    if not data:
        return None

    unique_elements = set(data)

    # Test the case of 'no mode'
    if len(unique_elements) == len(data):
        return 'All Elements'

    # A multi-map from a count to all the elements with that count
    count_to_element = {}

    for element in unique_elements:
        count = data.count(element)

        if count not in count_to_element:
            count_to_element[count] = []

        count_to_element[count].append(element)

    list_mode = count_to_element[max(count_to_element.keys())]

    if len(list_mode) == 1:
        return list_mode[0]

    else:
        list_mode.sort()
        return list_mode


@cache
def variance(data):
    """
    Calculate the variance of the data. The sum of the squares of the differences from the mean of the data,
    divided by the size of the data (as we're computing the sample variance, rather than the population one)
    :param data: An iterable containing numeric data
    :return: The variance of the data, or None if received an empty list
    """
    if not data:
        return None

    data_mean = mean(data)

    return sum([(x - data_mean) ** 2 for x in data]) / len(data)


def standard_deviation(data):
    """
    Calculate the standard deviation of the data - the square root of the variance. As our variance is the sample
    variance, the standard deviation will be the sample metric as well.
    :param data: An iterable containing numeric data
    :return: The standard deviation of the data, or None if received an empty list
    """
    if not data:
        return None

    return variance(data) ** 0.5


@cache
def linear_correlation(first_data, second_data):
    """
    Calculate the linear correlation between two sets of data. If either data set is empty, or they're of unequal side,
    return None. Otherwise, returns the sum of the products of the differences of each data point from its respective
    means, divided by the standard deviations and the length of the data (as we're computing the sample metric)
    :param first_data: An iterable containing numeric data
    :param second_data: An iterable containing numeric data
    :return: The correlation coefficient between the two data sets, or None of either list is empty or the lists are
    mismatched
    """
    if not first_data or not second_data or len(first_data) != len(second_data):
        return None

    zipped_data = zip(first_data, second_data)
    means = (mean(first_data), mean(second_data))
    std_devs = (standard_deviation(first_data), standard_deviation(second_data))

    return sum([(point[0] - means[0]) * (point[1] - means[1]) for point in zipped_data]) / \
        (std_devs[0] * std_devs[1] * len(zipped_data))


# -- Printing Functions -- #

def process_data_to_dict(all_data):
    """
    :param all_data: All data points, as a list of tuples, all of equal length. The first should contain the name
      of each data set (specific index within each tuple), the next should contain the measurements.
    :return data_dict: A dictionary from each variable name to the respective data set
    """
    names_tuple = all_data[0]
    data_without_names = all_data[1:]

    data_dict = {names_tuple[index]: tuple([item[index] for item in data_without_names])
                 for index in xrange(len(names_tuple))}
    # The previous line is equivalent to the following three
    # data_dict = {}
    # for index in xrange(len(names_tuple)):
    #     data_dict[names_tuple[index]] = tuple([item[index] for item in data_without_names])

    return data_dict


def print_table(header, data):
    """
    Print a table with a header row and data, using tabulate if possible
    :param header: The row of headers to print at the top of the table
    :param data:  The data to print, each row the same length as the header row
    :return: None. Data printed to output.
    """
    if tabulate:
        print tabulate.tabulate(data, header)

    else:
        for row in [header] + data:
            print '\t\t'.join([str(item) for item in row])

    # Blank line after each table
    print


ORDERED_FIELDS = [MEAN_FIELD, MEDIAN_FIELD, MODE_FIELD, VARIANCE_FIELD, STD_DEV_FIELD]
FIELDS_TO_FUNCTIONS = {MEAN_FIELD: mean, MEDIAN_FIELD: median, MODE_FIELD: mode, VARIANCE_FIELD: variance,
                       STD_DEV_FIELD: standard_deviation}


def print_descriptive_statistics(data_dict):
    """
    Prints a table with the descriptive statistics of all data sets given
    :param data_dict: A dictionary from each variable name to the respective data set
    :return: None, prints the results to the screen
    """

    # A dictionary from the name of each statistic to calculate to the values of the statistic, ordered like names_tuple
    stats_dict = {}
    names = data_dict.keys()
    for stat in ORDERED_FIELDS:
        stats_dict[stat] = [FIELDS_TO_FUNCTIONS[stat](data_dict[variable_name]) for variable_name in names]

    header_row = [NAME_FIELD] + ORDERED_FIELDS
    data_rows = []
    for index in xrange(len(names)):
        data_rows.append([names[index]] + [stats_dict[stat][index] for stat in ORDERED_FIELDS])

    data_rows.sort(key=lambda x: str.lower(x[0]))

    print_table(header_row, data_rows)


CORRELATION_HEADER_ROW = ['First Variable', 'Second Variable', "Pearson's r"]


def print_correlation_table(data_dict, test_func=None, test_threshold=DEFAULT_TEST_THRESHOLD):
    """
    Print the entire correlations, ordered by magnitude.
    If the parameter test_function is passed, compare the result from my function to the test function.
    :param data_dict: A dictionary from each variable name to the respective data set
    :return: None, prints the results to the screen.
    """
    variable_names = data_dict.keys()
    num_variables = len(variable_names)
    correlations = {}

    test_data = {}
    diffs = {}

    for first_index in xrange(num_variables - 1):
        for second_index in xrange(first_index + 1, num_variables):
            first_var = variable_names[first_index]
            first_data = data_dict[first_var]
            second_var = variable_names[second_index]
            second_data = data_dict[second_var]

            r = linear_correlation(first_data, second_data)
            correlations[r] = (first_var, second_var)

            if test_func:
                test_r = test_func(first_data, second_data)
                test_data[r] = test_r
                diffs[r] = abs(r - test_r)

    # Not in test mode
    if not test_func:
        correlation_rows = [(correlations[r][0], correlations[r][1], r) for r in
                            sorted(correlations, key=abs, reverse=True)]
        print_table(CORRELATION_HEADER_ROW, correlation_rows)

    # Test mode
    else:
        max_diff = max(diffs.values())

        if max_diff > test_threshold:
            print 'Correlation test failed! Max difference = {max_diff} > threshold = {threshold}'.format(
                max_diff=max_diff, threshold=test_threshold)

        else:
            print 'Correlation test passed! Max difference = {max_diff} <= threshold = {threshold}'.format(
                max_diff=max_diff, threshold=test_threshold)

        correlation_rows = [(correlations[r][0], correlations[r][1], r, diffs[r]) for r in
                            sorted(correlations, key=abs, reverse=True)]
        print_table(CORRELATION_HEADER_ROW + ['Difference'], correlation_rows)


def aggregate_stats(all_data=DEFAULT_DATA):
    """
    A wrapper function with the signature the assignment requests
    :param all_data: All data points, as a list of tuples, all of equal length. The first should contain the name
      of each data set (specific index within each tuple), the next should contain the measurements. Defaults to
      DEFAULT_DATA.
    :return: Nothing. Output printed to stdout.
    """
    data_dict = process_data_to_dict(all_data)
    print_descriptive_statistics(data_dict)
    print_correlation_table(data_dict)


# -- Test Functions -- #

def test_function(test_name, my_function, library_function, data_dict, threshold=DEFAULT_TEST_THRESHOLD):
    """
    Test the values produced by one of my functions to the matching library functions, or any two functions passed
    :param test_name: The name of the test to print
    :param my_function: My function to test
    :param library_function: The library function to compare it to
    :param data_dict: The dictionary of data to test
    :param threshold: The threshold above which the test fails, defaults to DEFAULT_TEST_THRESHOLD (10 ** -10)
    :return: True if the test passed, False if it failed
    """
    diffs = {}
    data_rows = []
    for variable_name in data_dict:
        variable_data = data_dict[variable_name]
        my_value = my_function(variable_data)
        library_value = library_function(variable_data)
        diff = abs(my_value - library_value)

        diffs[variable_name] = diff
        data_rows.append((variable_name, my_value, library_value, diff))

    max_diff = max(diffs.values())
    success = max_diff <= threshold

    if success:
        print '{name} test passed! Max difference = {max_diff} <= threshold = {threshold}'.format(
            name=test_name, max_diff=max_diff, threshold=threshold)

    else:
        print '{name} test failed! Max difference = {max_diff} > threshold = {threshold}'.format(
            name=test_name, max_diff=max_diff, threshold=threshold)

    header_row = (NAME_FIELD, 'My Value', 'Library Value', 'Difference')
    print_table(header_row, data_rows)

    return success


def wrapped_scipy_correlation(first_data, second_data):
    """
    scipy.stats.pearsonr returns a tuple of (r, 2-sided p-value). Since we only care about the first value,
    we wrap it quickly
    :param first_data: An iterable containing numeric data
    :param second_data: An iterable containing numeric data
    :return: The correlation coefficient (using scipy) between the two data sets,
    or None of either list is empty or the lists are mismatched
    """
    if not first_data or not second_data or len(first_data) != len(second_data):
        return None

    return scipy.stats.pearsonr(first_data, second_data)[0]


TEST_FIELDS_TO_FUNCTIONS = {MEAN_FIELD: (mean, numpy.mean),
                            MEDIAN_FIELD: (median, numpy.median),
                            VARIANCE_FIELD: (variance, numpy.var),
                            STD_DEV_FIELD: (standard_deviation, numpy.std)}


def library_test_suite(data_dict, threshold=DEFAULT_TEST_THRESHOLD):
    """
    A test suite comparing my calculated values to numpy and scipy's.
    Note: ignoring the mode, as I return all values and scipy's implementation only returns the first.
    :param data_dict: The dictionary from names to series
    :return: None. Prints test results to output.
    """
    if not (numpy and scipy):
        print 'Test suite requires numpy and scipy to run. Aborting...'
        return

    for function_name in TEST_FIELDS_TO_FUNCTIONS:
        functions = TEST_FIELDS_TO_FUNCTIONS[function_name]
        test_function(function_name, functions[0], functions[1], data_dict, threshold)

    print_correlation_table(data_dict, wrapped_scipy_correlation, threshold)


# -- Graphing Functions -- #

def validate_name_from_user(name, valid_names):
    """
    Try to interpret which of the data series in data_dict the user asked for by name
    :param name: The name of the series the user asked for
    :param valid_names: The valid series names
    :return: The appropriate series, recursing if not found
    """
    options = [series for series in valid_names if name in series]

    if not options:
        new_name = raw_input('{name} does not match any of the options: {all_names}'
                             .format(name=name, all_names=valid_names))
        validate_name_from_user(new_name, valid_names)

    if len(options) == 1:
        return options[0]

    else:
        new_name = raw_input('{name} matches multiple options: {options}. Which one did you intend? '
                             .format(name=name, options=options))
        return validate_name_from_user(new_name, valid_names)


def linear_regression(x_series, y_series, should_test=True, threshold=DEFAULT_TEST_THRESHOLD):
    """
    Seeing as we're in the theme of implementing everything ourselves, this is implementing linear regression,
    for shits and giggles
    :param x_series: The x-data
    :param y_series: The matching y-data
    :param should_test: If it should run a comparison test to scipy's linear regression as well. Default True.
    :return: m, b, r, r^2
    """
    x_mean = mean(x_series)
    y_mean = mean(y_series)

    x_std_dev = standard_deviation(x_series)
    y_std_dev = standard_deviation(y_series)

    r = linear_correlation(x_series, y_series)
    r_squared = r ** 2

    m = r * y_std_dev / x_std_dev
    b = y_mean - m * x_mean

    # Testing with standard library
    if scipy and should_test:
        scipy_regression = scipy.stats.linregress(x_series, y_series)
        data_rows = [['m', m, scipy_regression.slope],
                     ['b', b, scipy_regression.intercept],
                     ['r', r, scipy_regression.rvalue],
                     ['r^2', r_squared, scipy_regression.rvalue ** 2]]

        data_rows = [row + [abs(row[1] - row[2])] for row in data_rows]

        max_diff = max([row[3] for row in data_rows])
        success = max_diff < threshold

        if success:
            print 'Correlation test passed!\nMax difference = {max_diff} <= threshold = {threshold}'.format(
                max_diff=max_diff, threshold=threshold)

        else:
            print 'Correlation test failed! Max difference = {max_diff} > threshold = {threshold}'.format(
                max_diff=max_diff, threshold=threshold)

        header_row = (NAME_FIELD, 'My Value', 'Library Value', 'Difference')
        print_table(header_row, data_rows)

    return m, b, r, r_squared


MARKER_COLORS = ('red', 'blue')
LINE_COLORS = ('#CF5300', '#74BBFB')
LEGEND_LOCATIONS = ('upper left', 'upper right')
DEFAULT_HORIZONTAL_SIZE = 12
DEFAULT_VERTICAL_SIZE = 8
DEFAULT_DPI = 400


def draw_scatter(x_name, y_name, data_dict, axes, second=False):
    """
    Draws a single scatter plot - data and regression line. Expects the caller to do some of the general
    plot setup.
    :param x_name: The name of the X data series. Must be a valid key in data_dict - see scatter_plot for example usage.
    :param y_name: The name of the Y data series. Must be a valid key in data_dict - see scatter_plot for example usage.
    :param data_dict: The dictionary from data series name to values.
    :param axes: The axes generated by matplotlib - see scatter_plot for example usage.
    :param second: True if this is the second set of data to graph - False if it's the first.
    :return: None, set up a graph.
    """
    index = 0 + int(second)
    if second:
        axes = axes.twinx()

    x_series = data_dict[x_name]
    y_series = data_dict[y_name]
    m, b, r, r_squared = linear_regression(x_series, y_series)

    axes.plot(x_series, y_series, 'o', color=MARKER_COLORS[index], markersize=4, alpha=0.75)

    regression_label = 'y = {m:.2e}x + {b:.2e}\nr = {r:.2f}    r^2 = {r2:.2f}' \
        .format(m=m, b=b, r=r, r2=r_squared)
    x_for_regression = numpy.linspace(min(x_series), max(x_series), len(x_series))
    axes.plot(x_for_regression, m * numpy.array(x_for_regression) + b, '--', color=LINE_COLORS[index],
              label=regression_label, linewidth=3)

    axes.set_xlabel(x_name)
    axes.set_ylabel(y_name, color=MARKER_COLORS[index])
    axes.legend(loc=LEGEND_LOCATIONS[index])
    axes.grid(True, color=MARKER_COLORS[index])

    for tick in axes.get_yticklabels():
        tick.set_color(MARKER_COLORS[index])


def scatter_plot(x_series_name, y_series_name, second_y_series_name=None, all_data=DEFAULT_DATA):
    """
    Draws a scatter plot.
    :param x_series_name: A human-input name for the X series data. Validated using validate_name_from_user.
    :param y_series_name: A human-input name for the Y series data. Validated using validate_name_from_user.
    :param second_y_series_name: An optional name for a second Y series.
    :param all_data: All data points, as a list of tuples, all of equal length. The first should contain the name
      of each data set (specific index within each tuple), the next should contain the measurements.
    :return: None. Plots and shows a graph.
    """
    if not plot:
        print 'Scatter plot required matplotlib to run. Aborting...'
        return

    data_dict = process_data_to_dict(all_data)
    names = data_dict.keys()
    x_name = validate_name_from_user(x_series_name, names)
    y_name = validate_name_from_user(y_series_name, names)

    figure, axes = plot.subplots()
    figure.set_size_inches(DEFAULT_HORIZONTAL_SIZE, DEFAULT_VERTICAL_SIZE)
    figure.set_dpi(DEFAULT_DPI)

    draw_scatter(x_name, y_name, data_dict, axes)
    if second_y_series_name:
        second_y_name = validate_name_from_user(second_y_series_name, names)
        if second_y_name != y_name:
            draw_scatter(x_name, second_y_name, data_dict, axes, True)

    plot.show()


def histogram(series_name, all_data=DEFAULT_DATA):
    """
    Draws the histogram of a data series.
    :param series_name: The human-input name of a data series.  Validated using validate_name_from_user.
    :param all_data: All data points, as a list of tuples, all of equal length. The first should contain the name
      of each data set (specific index within each tuple), the next should contain the measurements.
    :return: None. Plots and shows the histogram of the data.
    """

    if not plot:
        print 'Histogram requires matplotlib to run. Aborting...'
        return

    data_dict = process_data_to_dict(all_data)
    name = validate_name_from_user(series_name, data_dict.keys())
    series = data_dict[name]

    plot.figure(figsize=(DEFAULT_HORIZONTAL_SIZE, DEFAULT_VERTICAL_SIZE), dpi=DEFAULT_DPI)
    plot.hist([series], color=[LINE_COLORS[0]], alpha=0.75)
    series_mean = mean(series)
    series_stddev = standard_deviation(series)
    label = '{name}\n mean = {mean:.2e}\nstddev = {stddev:.2e}' \
        .format(name=name, mean=series_mean, stddev=series_stddev)

    plot.legend([label], loc='best')
    plot.xlabel(name)
    plot.ylabel('Count')
    plot.grid(True)
    plot.show()


# -- Clustering Functions -- #

def kmeans_clustered_statistics(num_clusters, print_stats=True, print_correlations=True, all_data=DEFAULT_DATA):
    """
    Performs K-Means clustering and prints the descriptive statistics or correlations for each cluster separately
    :param num_clusters: How many clusters to sort the data into
    :param print_stats: Should the function print the descriptive statistics of each cluster. Default True.
    :param print_correlations: Should the function print the correlation table for each cluster. Default True.
    :param all_data: All data points, as a list of tuples, all of equal length. The first should contain the name
      of each data set (specific index within each tuple), the next should contain the measurements.
    :return:
    """
    if not sklearn:
        print 'Clustering requires sklearn. Aborting...'
        return

    if not print_stats and not print_correlations:
        print 'Must print statistics, correlation, or both. Aborting...'
        return

    print 'Clustering to {num_clusters} clusters...\n'.format(num_clusters=num_clusters)

    variable_names = all_data[0]
    variable_data = all_data[1:]

    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters)
    kmeans.fit(variable_data)
    cluster_centers = kmeans.cluster_centers_
    fitted_predictions = kmeans.predict(variable_data)
    cluster_to_data = {}

    for index in xrange(len(fitted_predictions)):
        cluster = fitted_predictions[index]

        if cluster not in cluster_to_data:
            cluster_to_data[cluster] = []

        cluster_to_data[cluster].append(variable_data[index])

    for cluster_index in xrange(len(cluster_centers)):
        cluster_data = cluster_to_data[cluster_index]
        cluster_data_dict = process_data_to_dict([variable_names] + cluster_data)

        print 'Printing cluster #{cluster_index} with {num} elements centered around:' \
            .format(cluster_index=cluster_index, num=len(cluster_to_data[cluster_index]))
        print '({center})'.format(center=', '.join(['{name} = {value:.2e}'
                                                   .format(name=name, value=value) for name, value in
                                                    zip(variable_names, cluster_centers[cluster_index])]))

        if print_stats:
            print_descriptive_statistics(cluster_data_dict)

        if print_correlations:
            print_correlation_table(cluster_data_dict)


# -- Main -- #

def main():
    aggregate_stats(DEFAULT_DATA)
    library_test_suite(process_data_to_dict(DEFAULT_DATA))

    scatter_plot('temperature', 'heating', 'cooling')
    kmeans_clustered_statistics(3)


if __name__ == '__main__':
    main()
