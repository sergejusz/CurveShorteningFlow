import math
import scipy.signal as signalproc

def pad_list(data, n):
    """
    Performs list padding at the beginning and at the end.
    Such kind of padding is helpful when median filter is used.
    Example: if n=3 and list contains 4 elements: [1, 2, 3, 4] then
    padded list should look like: [3, 4, 1, 2, 3, 4, 1, 2], so median filter
    will process list like circular list where beginning of the list
    comes after last element.
    :param data: list of values
    :param n: number of elements to pad.
    :return padded list.
    """
    new_data = []
    new_data.extend(data[-n:])
    new_data.extend(data)
    new_data.extend(data[0:n])
    return new_data

def shift_list(data, pos):
    """
    Shifts the list to start from specified position.
    Example: if list contains 3 elements: [1, 2, 3] and pos=1 then
    function will return list: [2, 3, 1].
    :param data: list of values
    :param pos: position of item to start from.
    :return shifted list.
    """
    new_data = []
    new_data.extend(data[pos:])
    new_data.extend(data[0:pos])
    return new_data

# calculates distance between two lists of the same length
# as a sum of absolute values of differences between elements : sum|a-b|
def distance_lists(a, b):
    """
    Returns distance between two lists of the same length.
    Distance is calculated as sum of absolute difference between elements of lists.
    :param a: first list.
    :param b: second list.
    :return distance between two given lists.
    """
    n = min(len(a), len(b))
    dist = 0
    for i in range(n):
        dist += math.fabs(a[i] - b[i])
    return dist

def med_filter_circular_list(values, kernel_size, iterations=1, error=0.0):
    """
    Performs repetitive median filtering of given list, processing it like circular.
    It means that first element goes immediately after last element (kind of 'closed' chain).
    For example, list of curvatures of closed plane curve could be the example of such list.
    :param values: source list of values.
    :param kernel_size: Size of median filter kernel (window). Should be odd number >= 3.
    :param iterations: Number of median fileter iterations.
    :param error: Parameter that is used to stop iterations when difference between list
    and filtered list becomes <= error (median filtering doesn't change source list).
    :return filtered list.
    """

    if kernel_size < 3 and (kernel_size % 2) == 0:
        return values
    k2 = kernel_size // 2
    source = pad_list(values, k2)
    for iter in range(iterations):
        tmp = signalproc.medfilt(source, kernel_size)
        dist = distance_lists(source[k2:-k2], tmp[k2:-k2])
        if dist <= error:
            return tmp[k2:-k2]
        source = pad_list(tmp[k2:-k2], k2)
    return source[k2:-k2]


def median_value(values):
    """
    Returns median of values from given list.
    :param values: source list of values.
    :return median of values in given list.
    """

    tmp = [v for v in values]
    tmp.sort()
    N = len(tmp)
    if (N % 2) == 0: 
        return (tmp[N//2]+tmp[(N//2)-1])*0.5 
    return tmp[N//2]

def mean_value(values):
    """
    Returns average of values from given list.
    :param values: list of values.
    :return average of values in given list.
    """
    return sum(values)/len(values)

# return position of maximal element
def argmax_list(values):
    """
    Returns position of maximum value in given list.
    :param values: list of values.
    :return position of maximum value in given list.
    """
    pos = -1
    max_val = min(values) - 1;
    for i in range(len(values)):
        if max_val < values[i]:
            pos = i
            max_val = values[i]
    return pos
