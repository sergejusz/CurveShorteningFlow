import math
import numpy as np
from scipy import signal
from scipy import interpolate
from numpy.polynomial import polynomial as poly
import list_operations as listops

def resample_curve(curve, num):
    resampled_curve = []
    if len(curve) == 0 or num == 0: return resampled_curve

    arclen = []
    arclen.append(0.0)
    p1 = curve[0]
    for i in range(1, len(curve)):
        p2 = curve[i]
        s = arclen[i-1] + math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        arclen.append(s)
        p1 = p2

    L = arclen[-1]
    step = L/num

    resampled_curve.append(curve[0])
    j = 0
    for i in range(1, num):
        dist = step*i
        
        while not (dist >= arclen[j] and dist <= arclen[j+1]):
            j +=1

        a = (dist - arclen[j])/(arclen[j+1]-arclen[j])
        p1 = curve[j]
        p2 = curve[j+1]
        x = a*p1[0] + (1.0-a)*p2[0]
        y = a*p1[1] + (1.0-a)*p2[1]
        resampled_curve.append((x, y))
    return resampled_curve


def get_curvature_xy(x, y, w=5, po=2):
    x_der1 = signal.savgol_filter(x, window_length=w, polyorder=po, deriv=1, mode="wrap")
    y_der1 = signal.savgol_filter(y, window_length=w, polyorder=po, deriv=1, mode="wrap")
    x_der2 = signal.savgol_filter(x, window_length=w, polyorder=po, deriv=2, mode="wrap")
    y_der2 = signal.savgol_filter(y, window_length=w, polyorder=po, deriv=2, mode="wrap")
    a = [y_der2[i]*x_der1[i]-x_der2[i]*y_der1[i] for i in range(0, len(x_der1))]
    b = [(math.hypot(x_der1[i], y_der1[i]))**3 for i in range(0, len(x_der1))]
    return [a[i]/b[i] for i in range(0, len(a))]

def get_curvature(curve, w=5, po=2):
    return get_curvature_xy([p[0] for p in curve], [p[1] for p in curve], w, po)
    
def get_tangent_field(curve, w=5, po=1):
    x_der1 = signal.savgol_filter([p[0] for p in curve], window_length=w, polyorder=po, deriv=1, mode="wrap")
    y_der1 = signal.savgol_filter([p[1] for p in curve], window_length=w, polyorder=po, deriv=1, mode="wrap")
    return [(x_der1[i],y_der1[i]) for i in range(0, len(x_der1))]

def get_normal_field(curve, w=3, po=1):
    der1 = get_tangent_field(curve, w, po)
    return [(d[1], -d[0]) for d in der1]

def get_normal_unit_field(curve, w=3, po=1):
    return normalize(get_normal_field(curve, w, po))
    
def normalize(vectors):
    normalized = []
    for v in vectors:
        h = math.hypot(v[0], v[1])
        normalized.append((v[0]/h if h > 0.0 else 0.0, v[1]/h if h > 0.0 else 0.0))
    return normalized

def smoothen_curve(curve, w=3, po=2, iter=1):
    if iter == 0:
        return curve
    x = [p[0] for p in curve]
    y = [p[1] for p in curve]
    while iter > 0:
        x1 = signal.savgol_filter(x, window_length=w, polyorder=po, mode="wrap")
        y1 = signal.savgol_filter(y, window_length=w, polyorder=po, mode="wrap")
        iter -= 1
        if iter > 0:
            x.clear()
            x.extend(x1)
            y.clear()
            y.extend(y1)
            
    return [(x1[i], y1[i]) for i in range(0, len(curve))]

def shift_curve(curve, index):
    n = len(curve)
    if n == 0:
        return []
    new_curve = []
    for i in range(0, n):
        new_curve.append(curve[(i + index) % n])
    return new_curve

def translate(curve, pt):
    return [(p[0] + pt[0], p[1] + pt[1]) for p in curve]
    
def translate2(curve, p):
    x = [p[0] for p in curve]
    y = [p[1] for p in curve]
    return [(x[i] + p[0], y[i] + p[1]) for i in range(0, len(curve))]

def move_curve_center(curve, new_center):
    curve_center = get_curve_center(curve)
    #rect_center = (width*0.5, height*0.5)
    v = (new_center[0] - curve_center[0], new_center[1] - curve_center[1])
    return translate(curve, v)

def get_curve_center(curve):
    n = len(curve)
    return (sum([p[0] for p in curve])/n, sum([p[1] for p in curve])/n)

def get_distances_to_point(pt, curve):
    distances = []
    for p in curve:
        distances.append(math.hypot(p[0]-pt[0], p[1]-pt[1]))
    return distances
    
def get_sum_distances_to_point(pt, curve):
    sum = 0.0
    for p in curve:
        sum += math.hypot(p[0]-pt[0], p[1]-pt[1])
    return sum

def get_mean_distances_to_point(pt, curve):
    sum = 0.0
    for p in curve:
        sum += math.hypot(p[0]-pt[0], p[1]-pt[1])
    return sum/len(curve)

def homothety_transform(curve, pc, alpha):
    new_curve = []
    for p in curve:
        new_curve.append(((p[0] - pc[0])*alpha + pc[0], (p[1] - pc[1])*alpha + pc[1]))
    return new_curve

def get_curve_length(curve):
    if len(curve) == 0: return 0.0
    n = len(curve)
    l = 0
    p1 = curve[0]
    for i in range(1, n):
        p2 = curve[i]
        l += math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        p1 = p2
    return l + math.hypot(curve[0][0]-curve[n-1][0], curve[0][1]-curve[n-1][1])

def get_curve_steps(curve):
    s = []
    if len(curve) == 0: return s
    n = len(curve)
    p1 = curve[0]
    for i in range(1, n):
        p2 = curve[i]
        s.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        p1 = p2
    return s

def get_curve_length_list(curve, wrap=False):
    s = []
    if len(curve) == 0: return s
    n = len(curve)
    l = 0
    s.append(l)
    p1 = curve[0]
    for i in range(1, n):
        p2 = curve[i]
        l += math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        s.append(l)
        p1 = p2
    if wrap:
        s.append(l + math.hypot(curve[0][0]-curve[n-1][0], curve[0][1]-curve[n-1][1]))
    return s

def get_part_curve_length(curve, i, j):
    n = len(curve)
    if n == 0: return 0
    if i == j: return 0
    l = 0;
    p1 = curve[i % n]
    for k in range(i+1, j+1):
        #print("k=", k)
        p2 = curve[k % n]
        l += math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        p1 = p2
    return l

def get_excl_curve_length(curve, groups):
    n = len(curve)
    if n == 0: return 0.0
    m = len(groups)
    l = get_curve_length(curve)
    if m == 0:
        return l

    s = 0
    for i in range(0, m):
        s += get_part_curve_length(curve, groups[i][0], groups[i][1])
    return l - s

    
def get_mean_curvature(curve, curvature):
    if len(curve) == 0: return 0.0
    sumc = 0
    L = 0
    p1 = curve[0]
    for i in range(1, len(curve)):
        p2 = curve[i]
        ds = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        sumc += ds*curvature[i-1]
        L += ds
        p1 = p2
    return sumc/L

def get_curvature_over_curve(curve, curvature):
    if len(curve) == 0: return 0.0
    sumc = 0
    p1 = curve[0]
    for i in range(1, len(curve)):
        p2 = curve[i]
        ds = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        sumc += ds*curvature[i-1]
        p1 = p2
    return sumc

def get_horizontal_amplitude(curve):
    if len(curve) == 0:
        return 0,0
    return (max(p[0] for p in curve) - min(p[0] for p in curve))
    
def get_vertical_amplitude(curve):
    if len(curve) == 0:
        return 0.0
    return (max(p[1] for p in curve) - min(p[1] for p in curve))

# very primitive estimation of radius if curve is circle
def get_radius_estimation(curve):
    if len(curve) == 0:
        return 0.0
    ampl_hor = get_horizontal_amplitude(curve)
    ampl_vert = get_vertical_amplitude(curve)
    return (ampl_hor + ampl_vert)*0.25

def is_circle(curve):
    if len(curve) == 0:
        return False
    pc = get_curve_center(curve)
    distances = get_distances_to_point(pc, curve)
    radius_estimated = listops.median_value(distances)
    s = get_curve_length(curve)
    radius = s/(2.0*np.pi)
    # relative tolerance threshold 0.5%
    threshold = 0.5
    #print("p=", 100.0*(math.fabs(radius_estimated - radius)/radius_estimated))
    return 100.0*(math.fabs(radius_estimated - radius)/radius_estimated) <= threshold

def resample_by_lsq(curve, w=7, polyorder=2, n=-1):
    nc = len(curve)
    if nc == 0:
        return []
    np = n if n != -1 else nc
    new_curve = []
    s = get_curve_length(curve)
    ds = s/np
    w2 = w // 2
    sp = get_curve_length_list(curve)
    dist_mean = []
    
    i = 0
    for k in range(0, np):
        sk = ds*k
        if k == 0:
            i = 0
        else:
            while i > 0 and i < nc and not(sk <= sp[i] and sk > sp[i-1]):
                i += 1
            if i == nc:
                if sk > sp[nc-1] and sk <= s:
                    i = nc
        t = []
        x = []
        y = []
        s0 = 0
        for j in range(i-w2, i+w2+1):
            l = j if j<0 else j%nc
            if j>i-w2 and s0<=0:
                if t[j-i+w2-1] > sp[l]: s0 = s
            t.append(sp[l] + s0)
            x.append(curve[l][0])
            y.append(curve[l][1])
        if sk < t[0]: sk += s0
        p1 = poly.polyfit(t, x, polyorder)
        x1 = poly.polyval(sk, p1)
        p2 = poly.polyfit(t, y, polyorder)
        y1 = poly.polyval(sk, p2)
        if len(new_curve) > 0:
            mm = len(new_curve)
            dist_mean.append(math.hypot(new_curve[mm-1][0]-x1, new_curve[mm-1][1]-y1))
        new_curve.append((x1, y1))
        if i == 0: i = 1
    return new_curve


def resample_by_interpolation(curve):
    s = get_curve_length(curve)
    n = len(curve)
    # current discretization
    length_list = get_curve_length_list(curve, wrap=True)
    fx = interpolate.interp1d(length_list, [p[0] for p in curve] + [curve[0][0]], 'cubic')
    fy = interpolate.interp1d(length_list, [p[1] for p in curve] + [curve[0][1]], 'cubic')
    # uniform discretization
    t = [(s/n)*i for i in range(0, n)]
    x = fx(t)
    y = fy(t)
    return [(x[i], y[i]) for i in range(0, len(x))]
