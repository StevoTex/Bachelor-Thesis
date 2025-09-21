def METRIC_SUM_WITH_OPTIONAL_POWER(*values, **kwargs):
    """
    Calculates the sum of the given values. The power of each value can be set using the keyword argument 'pow'.
    :param values: The values.
    :type values: float or int
    :param kwargs: Keyword arguments.
    :return: The sum of the values.
    :rtype: float or int
    """
    power = kwargs.get('pow', 1)
    return sum([v ** power for v in values])


def METRIC_SUM(*values):
    """
    Calculates the sum of the given values.
    :param values: The values.
    :type values: float or int
    :return: The squared sum of the values.
    :rtype: float or int
    """
    return METRIC_SUM_WITH_OPTIONAL_POWER(*values)


def METRIC_SQUARED_SUM(*values):
    """
    Calculates the squared sum of the given values.
    :param values: The values.
    :type values: float or int
    :return: The squared sum of the values.
    :rtype: float or int
    """
    return METRIC_SUM_WITH_OPTIONAL_POWER(*values, pow=2)


def METRIC_AVG(*values):
    """
    Calculates the average of the given values.
    :param values: The values.
    :type values: float or int
    :return: The average of the values.
    :rtype: float or int
    """
    count = len(values)
    if count == 0:
        return 0
    return sum(values) / count


def METRIC_MAX(*values):
    """
    Calculates the maximum of the given values.
    :param values: The values.
    :type values: float or int
    :return: The maximum of the values.
    :rtype: float or int
    """
    return max(values)


def METRIC_MIN(*values):
    """
    Calculates the minimum of the given values.
    :param values: The values.
    :type values: float or int
    :return: The minimum of the values.
    :rtype: float or int
    """
    return min(values)