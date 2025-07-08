import math
import numpy as np
from scipy import signal
from scipy import interpolate
from numpy.polynomial import polynomial as poly
import list_operations as list_ops
import curve_operations as curve_ops
        
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


def get_curvature(curve, w=5, po=2):
    x_der1 = signal.savgol_filter(curve[0], window_length=w, polyorder=po, deriv=1, mode="wrap")
    y_der1 = signal.savgol_filter(curve[1], window_length=w, polyorder=po, deriv=1, mode="wrap")
    x_der2 = signal.savgol_filter(curve[0], window_length=w, polyorder=po, deriv=2, mode="wrap")
    y_der2 = signal.savgol_filter(curve[1], window_length=w, polyorder=po, deriv=2, mode="wrap")
    return np.divide(np.subtract(np.multiply(y_der2, x_der1), np.multiply(x_der2, y_der1)), np.power(np.hypot(x_der1, y_der1), 3))


def get_tangent_field(curve, w=5, po=1):
    return signal.savgol_filter(curve, window_length=w, polyorder=po, deriv=1, mode="wrap")

   
def get_normal_field(curve, w=3, po=1):
    der1 = get_tangent_field(curve, w, po)
    return np.array([der1[1], -der1[0]])

def get_normal_unit_field(curve, w=3, po=1):
    return normalize(get_normal_field(curve, w, po))
    
def normalize(vectors):
    h = np.hypot(vectors[0],vectors[1])
    return np.divide(vectors, np.add(h, np.fabs(np.subtract(np.sign(h), 1.))))

def smoothen_curve(curve, w=3, po=2, iter=1):
    if iter == 0:
        return curve
    for i in range(0, iter):
        curve = signal.savgol_filter(curve, window_length=w, polyorder=po, mode="wrap")
    return curve

def shift_curve(curve, index):
    n = curve.shape[1:][0]
    if n == 0:
        return get_empty_curve()
    idx1 = [(i+index)%n for i in range(0,n)]
    idx2 = [n + (i+index)%n for i in range(0,n)]
    return np.take(curve, [idx1, idx2])


def translate(curve, x, y):
    return np.add(curve, ([x], [y]))


def move_curve_center(curve, x, y):
    [cx, cy] = get_curve_center(curve)
    return translate(curve, x - cx, y - cy)

def get_curve_center(curve):
    return np.divide(np.sum(curve, axis=1), curve.shape[1:][0])

def get_distances_to_point(x, y, curve):
    return np.hypot(np.subtract(curve[0], x), np.subtract(curve[1], y))

def get_sum_distances_to_point(x, y, curve):
    return np.sum(get_distances_to_point(x, y, curve))

def get_mean_distances_to_point(x, y, curve):
    if curve_ops.is_empty_curve(curve): return 0
    return get_sum_distances_to_point(x, y, curve)/curve_ops.get_curve_size(curve)


def homothety_transform(curve, x, y, alpha):
    return np.add(np.multiply(np.subtract(curve, ([x], [y])), alpha), ([x], [y]))

def get_curve_length(curve):
    if curve_ops.is_empty_curve(curve): return 0
    l = np.sum(np.hypot(np.subtract(curve[0][1:], curve[0][:-1]), np.subtract(curve[1][1:],curve[1][:-1]))) 
    return l + math.hypot(curve[0][0]-curve[0][-1], curve[1][0]-curve[1][-1])


def get_curve_steps(curve):
    return np.append(np.hypot(np.subtract(curve[0][1:], curve[0][:-1]), np.subtract(curve[1][1:],curve[1][:-1])), math.hypot(curve[0][0]-curve[0][-1], curve[1][0]-curve[1][-1]))


def get_curve_length_list(curve, wrap=False):
    return np.cumsum(np.append([0.0], get_curve_steps(curve)))


def get_part_curve_length(curve, i, j):
	if curve_ops.is_empty_curve(curve): return 0
	n = curve_ops.get_curve_size(curve)
	if i == j: return 0
	l = 0;
	x1 = curve[0][i % n]
	y1 = curve[1][i % n]
	for k in range(i+1, j+1):
		x2 = curve[0][k % n]
		y2 = curve[1][k % n]
		l += math.hypot(x2-x1, y2-y1)
		x1 = x2
		y1 = y2
	return l

def get_excl_curve_length(curve, groups):
	if curve_ops.is_empty_curve(curve): return 0
	n = curve_ops.get_curve_size(curve)
    
	m = len(groups)
	l = get_curve_length(curve)
	if m == 0:
		return l

	s = 0
	for i in range(0, m):
		s += get_part_curve_length(curve, groups[i][0], groups[i][1])
	return l - s

    
def get_curvature_over_curve(curve, curvature):
	if curve_ops.is_empty_curve(curve): return 0.0
	return np.sum(np.multiply(get_curve_steps(curve), curvature))

def get_horizontal_amplitude(curve):
	if curve_ops.is_empty_curve(curve): return 0.0
	return np.max(curve, axis=1)[0] - np.min(curve, axis=1)[0]
    
def get_vertical_amplitude(curve):
	if curve_ops.is_empty_curve(curve): return 0.0
	return np.max(curve, axis=1)[1] - np.min(curve, axis=1)[1]

# very primitive estimation of radius if curve is circle
def get_radius_estimation(curve):
	return (get_horizontal_amplitude(curve) + get_vertical_amplitude(curve))*0.25

def is_circle(curve):
	if curve_ops.is_empty_curve(curve): return False
	pc = get_curve_center(curve)
	distances = get_distances_to_point(pc[0], pc[1], curve)
	radius_estimated = list_ops.median_value(distances)
	s = get_curve_length(curve)
	radius = s/(2.0*np.pi)
	# relative tolerance threshold 0.5%
	threshold = 0.5
	#print("p=", 100.0*(math.fabs(radius_estimated - radius)/radius_estimated))
	return 100.0*(math.fabs(radius_estimated - radius)/radius_estimated) <= threshold

def resample_by_lsq(curve, w=7, polyorder=2, n=-1):
	if curve_ops.is_empty_curve(curve): return curve_ops.get_empty_curve()
	nc = curve_ops.get_curve_size(curve)
	m = n if n != -1 else nc
	x_vec = []
	y_vec = []
	s = get_curve_length(curve)
	ds = s/m
	w2 = w // 2
	sp = get_curve_length_list(curve)
    
	i = 0
	for k in range(0, m):
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
			x.append(curve[0][l])
			y.append(curve[1][l])
		if sk < t[0]: sk += s0
		p1 = poly.polyfit(t, x, polyorder)
		x1 = poly.polyval(sk, p1)
		p2 = poly.polyfit(t, y, polyorder)
		y1 = poly.polyval(sk, p2)
		x_vec.append(x1)
		y_vec.append(y1)
		if i == 0: i = 1
	return np.array([x_vec, y_vec])

def resample_by_interpolation(curve):
	s = get_curve_length(curve)
	n = curve_ops.get_curve_size(curve)
	# current discretization
	length_list = get_curve_length_list(curve, wrap=True)
	fx = interpolate.interp1d(length_list, np.append(curve, [[curve[0][0]], [curve[1][0]]], axis=1)[0], 'cubic')
	fy = interpolate.interp1d(length_list, np.append(curve, [[curve[0][0]], [curve[1][0]]], axis=1)[1], 'cubic')
	# uniform discretization
	t = np.linspace(0.0, s, n, endpoint=False)
	x = fx(t)
	y = fy(t)
	return np.array([fx(t), fy(t)])
