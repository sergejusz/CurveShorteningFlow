import math
import scipy.signal as signalproc

# pads list at the beginning and end to correctly apply median filter
def pad_list(data, n):
    new_data = []
    new_data.extend(data[-n:])
    new_data.extend(data)
    new_data.extend(data[0:n])
    return new_data

# shift the list to start from specified position
def shift_list(data, pos):
    new_data = []
    new_data.extend(data[pos:])
    new_data.extend(data[0:pos])
    return new_data

# calculates distance between two lists of the same length
# as a sum of absolute values of differences between elements : sum|a-b|
def distance_lists(a, b):
    n = min(len(a), len(b))
    dist = 0
    for i in range(n):
        dist += math.fabs(a[i] - b[i])
    return dist

# applies median filter for wrapped list
def med_filter_wrapped(values, kernel_size, iterations=1, error=0.0):
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

# returns median of list
def median_value(values):
    tmp = [v for v in values]
    tmp.sort()
    N = len(tmp)
    if (N % 2) == 0: 
        return (tmp[N//2]+tmp[(N//2)-1])*0.5 
    return tmp[N//2]

def mean_value(values):
    return sum(values)/len(values)

# return position of maximal element
def argmax_list(data):
    pos = -1
    max_val = min(data) - 1;
    for i in range(len(data)):
        if max_val < data[i]:
            pos = i
            max_val = data[i]
    return pos
