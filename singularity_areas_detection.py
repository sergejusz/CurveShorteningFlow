import math
import list_operations as listops


def extract_groups(data):
    if len(data) == 0:
        return []
    threshold = sum(data)/len(data)

    groups = []    
    while True:
        max_pos = listops.argmax_list(data)
        if max_pos == -1:
            return []
        max_val = data[max_pos]
        if max_val > threshold:
            a = max_pos
            for i in range(max_pos, -1, -1):
                if data[i] > threshold:
                    a = i
                    data[i] = threshold - 1
                else:
                    break
            b = max_pos + 1
            for i in range(max_pos + 1, len(data)):
                if data[i] > threshold:
                    b = i
                    data[i] = threshold - 1
                else:
                    break
            groups.append((a, b))
        else:
            break
    return groups

# returns True if curve points from a to b (indexes)
# are singular - if there are curve points 
# with positive and negative curvature
def is_singular(data, a, b):
    num_positive = 0
    num_negative = 0
    n = len(data)
    i = a
    while i <= b:
        if data[i % n] < 0:
            num_negative += 1
        else:
            num_positive += 1
        i += 1
    #print("neg=", num_negative, " pos=", num_positive)
    return num_negative > 1 and num_positive > 1

def join_groups(groups, n):
    processed = []
    processed.extend(groups)
    m = len(groups)
    if m < 2:
        return processed
    a = groups[0]
    b = groups[m-1]
    if a[0] == 0 and b[1] == n-1:
        g = (b[0], a[1] + n)
        processed.pop(0)
        processed.pop(m-2)
        processed.append(g)
    return processed

        
def binarize(values):
    abs_values = [math.fabs(v) for v in values]
    max_val = max(abs_values)
    mean_val = listops.mean_value(abs_values)
    positions = [i for i in range(len(abs_values)) if abs_values[i] >= mean_val]
    tmp = [0.0]*len(values)
    for pos in positions:
        tmp[pos] = abs_values[pos]
    tmp2 = listops.med_filter_wrapped(tmp, 5, 100)
    return tmp2


def detect(values):
    max_val = max(values)
    groups = extract_groups(binarize(values))
    singular_areas = []
    if len(groups) > 0:
        tmp = [0]*len(values)
        for g in groups:
             i = g[0]
             if is_singular(values, g[0], g[1]):
                singular_areas.append(g)
        if len(singular_areas) > 0:
            singular_areas.sort(key=lambda tup: tup[0])
            joined = join_groups(singular_areas, len(values))
            #print("n=", len(values), " noisy groups=", singular_areas, " joined groups=", joined)
            return joined
    return []
