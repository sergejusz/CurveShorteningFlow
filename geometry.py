import math
import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.spatial import ConvexHull
from numpy.polynomial import polynomial as poly
import list_operations as list_ops

#pylint: disable=line-too-long

def get_curve_size(curve : np.ndarray) -> int :
    """
    Returns number of points in curve.
    :param curve: plane curve.
    :return number of points in curve.
    """
    return curve.shape[1:][0]


def is_empty_curve(curve : np.ndarray) -> bool:
    """
    Returns True if curve has no points.
    :param curve: plane curve.
    :return True if curve has no points, False otherwise.
    """
    return curve.size == 0


def get_empty_curve() -> np.ndarray :
    """
    Returns empty curve.
    :return empty curve.
    """
    return np.array([[], []])


def get_ellipse(cx : float, cy : float, radiusx : float, radiusy : float, num_points: int) -> np.ndarray :
    """
    Returns ellipse curve with given parameters.
    :param cx: x-coordinate of center of ellipse.
    :param cy: y-coordinate of center of ellipse.
    :param radiusx: horizontal radius of ellipse.
    :param radiusy: vertical radius of ellipse.
    :param num_points: number of points in ellipse.
    :return ellipse curve.
    """
    t = np.linspace(0.0, 2.0*np.pi, num_points, endpoint=False)
    return np.array([cx + radiusx*np.cos(t), cy + radiusy*np.sin(t)])

def get_circle(cx : float, cy : float, radius : float, num_points : int) -> np.ndarray :
    """
    Returns circle curve with given parameters.
    :param cx: x-coordinate of center of circle.
    :param cy: y-coordinate of center of circle.
    :param radius: radius of circle.
    :param num_points: number of points in circle.
    :return circle curve.
    """
    return get_ellipse(cx, cy, radius, radius, num_points)

def get_paperclip(cx : float, cy : float, radius : float, num_points : int ) -> np.ndarray :
    """
    Returns paperclip shaped curve with given parameters.
    :param cx: x-coordinate of center of curve.
    :param cy: y-coordinate of center of curve.
    :param radius: radius of circle that is part of paperclip curve.
    :param num_points: number of points in curve.
    :return paperclip shaped curve.
    """
    # number of points for circular areas
    n1 = int((np.pi * num_points)/(3 + np.pi))
    # number of points for straight sides
    n2 = num_points - n1
    t1 = np.linspace(1.5 * np.pi, 0.5 * np.pi, n1 // 2, endpoint=False)
    z1 = np.array([radius * np.cos(t1), radius * np.sin(t1)])
    z2 = np.array([np.linspace(0.0, 3.0*radius, n2 // 2, endpoint=False), np.full(n2 // 2, radius)])
    t3 = np.linspace(2.5 * np.pi, 1.5 * np.pi, n1 - ( n1 // 2), endpoint=False)
    z3 = np.array([3.0 * radius + radius * np.cos(t3), radius * np.sin(t3)])
    z4 = np.array([np.linspace(3.0*radius, 0.0, n2 - (n2 // 2), endpoint=False), np.full(n2 - (n2 // 2), -radius)])
    return np.add(np.append(np.append(np.append(z1, z2, axis=1), z3, axis=1), z4, axis=1), [[cx], [cy]])

def get_rectangle(cx : float, cy : float, a : float, b : float, num_points : int) -> np.ndarray :
    """
    Returns rectangle shaped curve with given parameters.
    :param cx: x-coordinate of center of curve.
    :param cy: y-coordinate of center of curve.
    :param a: horizontal side of rectangle.
    :param b: vertical side of rectangle
    :param num_points: number of points in curve.
    :return rectangle shaped curve.
    """
    t = np.linspace(0.0, a + a + b + b, num_points, endpoint=False)
    t1 = [_ for _ in t if _ < a]
    p1 = np.array([t1, np.full(len(t1), 0.0)])
    t2 = [_ - a for _ in t if _ >= a and _ < a + b]
    p2 = np.array([np.full(len(t2), a), t2])
    t3 = [a + a + b - _ for _ in t if _ >= a + b and _ < a + a + b]
    p3 = np.array([t3, np.full(len(t3), b)])
    t4 = [a + a + b + b - _ for _ in t if _ >= a + a + b]
    p4 = np.array([np.full(len(t4), 0.0), t4])
    return np.add(np.append(np.append(np.append(p1, p2, axis=1), p3, axis=1), p4, axis=1), [[cx], [cy]])

def get_curvature(curve : np.ndarray, w : int=5, po : int=2) -> np.ndarray :
    """
    Calculates plane curve curvature.
    :param curve: input plane curve.
    :param w:  window size to calculate derivatives (Default value = 5)
    :param po:  polynomial order to calculate derivatives (Default value = 2)
    :return curvature array at each point of curve.
    """
    der1 = signal.savgol_filter(curve, window_length=w, polyorder=po, deriv=1, mode="wrap")
    der2 = signal.savgol_filter(curve, window_length=w, polyorder=po, deriv=2, mode="wrap")
    return np.divide(np.subtract(np.multiply(der2[1], der1[0]), np.multiply(der2[0], der1[1])),
                     np.power(np.hypot(der1[0], der1[1]), 3))


def get_tangent_field(curve : np.ndarray, w : int=3, po : int=1) -> np.ndarray :
    """
    Returns array of tangent vectors to curve at each point.
    :param curve: plane curve
    :param w:  window size to calculate derivatives (Default value = 3)
    :param po:  polynomial order to calculate derivatives (Default value = 1)
    :return array of tangent vectors.
    """
    return signal.savgol_filter(curve, window_length=w, polyorder=po, deriv=1, mode="wrap")


def get_normal_field(curve : np.ndarray, w: int=3, po: int=1) -> np.ndarray :
    """
    Returns array of normal vectors to curve at each point.
    :param curve: plane curve
    :param w:  window size to calculate derivatives (Default value = 3)
    :param po:  polynomial order to calculate derivatives (Default value = 1)
    :return array of normal vectors.
    """
    der1 = get_tangent_field(curve, w, po)
    return np.array([der1[1], -der1[0]])


def get_normal_unit_field(curve : np.ndarray, w : int=3, po : int=1) -> np.ndarray :
    """

    :param curve: 
    :param w:  (Default value = 3)
    :param po:  (Default value = 1)

    """
    return normalize(get_normal_field(curve, w, po))


def normalize(vectors : np.ndarray) -> np.ndarray:
    """
    Normalizes array of vectors (length=1).
    :param vectors: 
    :return normalized vectors
    """
    h = np.hypot(vectors[0], vectors[1])
    return np.divide(vectors, np.add(h, np.fabs(np.subtract(np.sign(h), 1.))))


def smoothen_curve(curve : np.ndarray, w : int=3, po : int=2, iterations : int=1) -> np.ndarray:
    """
    Iterative smoothing of plane curve using Savitzky-Golay filter.
    :param curve: plane curve.
    :param w:  (Default value = 3)
    :param po:  (Default value = 2)
    :param iterations:  (Default value = 1)
    :return smoothed curve.
    """
    if iterations == 0:
        return curve
    for _ in range(iterations):
        curve = signal.savgol_filter(curve, window_length=w, polyorder=po, mode="wrap")
    return curve


def smoothen_with_compensation_curve(curve : np.ndarray, w : int=3, po : int=2, iterations : int=1) -> np.ndarray:
    """
    Smoothing of plane curve with size compensation.
    Since smoothing shrinks curve homothety transformation is used
    to keep average distance from curve points to center of curve.
    :param curve: plane curve.
    :param w:  window size for smoothing (Default value = 3)
    :param po:  polynomial order for smoothing (Default value = 2)
    :param iterations:  number of smoothing iterations to apply to curve (Default value = 1)
    :return smoothed curve.
    """
    # get data that we need for compensation of shrinking effect of smoothing
    # for that we use average distance from curve points to the center of a curve
    cx, cy = get_curve_center(curve)
    r0 = get_mean_distances_to_point(cx, cy, curve)

    #perform smoothing of curve using Savitzky-Golay
    curve = smoothen_curve(curve, w, po, iterations)
    # we use homothety transformation to compensate that
    # curve slightly shrinks after smoothing
    r1 = get_mean_distances_to_point(cx, cy, curve)
    return homothety_transform(curve, cx, cy, r0 / r1)

def shift_curve(curve : np.ndarray, index : int) -> np.ndarray:
    """
    Changes starting point of curve.
    :param curve: plane curve
    :param index: position pointing to new curve beginning.
    :return curve
    """
    n = get_curve_size(curve)
    if n == 0:
        return get_empty_curve()
    idx1 = [(i + index) % n for i in range(n)]
    idx2 = [n + (i + index) % n for i in range(n)]
    return np.take(curve, [idx1, idx2])


def translate(curve : np.ndarray, x : float, y : float) -> np.ndarray:
    """
    Translates the whole curve along vector (x,y)
    :param curve: curve data.
    :param x: x-coordinate of translation vector.
    :param y: y-coordinate of translation vector.
    :return Translated curve.
    """
    return np.add(curve, ([x], [y]))


def move_curve_center(curve : np.ndarray, x : float, y : float) -> np.ndarray:
    """
    Moves curve center to new location.
    :param curve: curve data.
    :param x: x-coordinate of new center.
    :param y: y-coordinate of new center.
    :return Curve moved to new center.
    """
    [cx, cy] = get_curve_center(curve)
    return translate(curve, x - cx, y - cy)


def get_curve_center(curve : np.ndarray) -> np.ndarray:
    """
    Returns curve center.
    :param curve: plane curve.
    :return center of given curve.
    """
    return np.divide(np.sum(curve, axis=1), get_curve_size(curve))

def rotate_curve(curve : np.ndarray, fi : float) -> np.ndarray:
    """
    Rotates curve around its center for given angle in radians.
    :param curve: curve array
    :param fi: angle value in radians
    :return curve rotated by given fi.
    """
    [cx, cy] = get_curve_center(curve)
    cosfi = math.cos(fi)
    sinfi = math.sin(fi)
    return translate(np.dot(np.array([[cosfi, sinfi], [-sinfi, cosfi]]), translate(curve, -cx, -cy)), cx, cy)


def get_distances_to_point(x : float, y : float, curve  : np.ndarray) -> np.ndarray:
    """
    Calculates array of distance from given points to each point of curve.
    :param x: x-coordinate of point to get distances to.
    :param y: y-coordinate of point to get distances to
    :param curve: plane curve.
    :return distances to point.
    """
    return np.hypot(np.subtract(curve[0], x), np.subtract(curve[1], y))


def get_sum_distances_to_point(x : float, y : float, curve : np.ndarray) -> float:
    """
    Calculates sum of distances from given point to each point of curve.
    :param x: x-coordinate of given point
    :param y: y-coordinate of given point
    :param curve: plane curve
    :return sum of distances to point.
    """
    return np.sum(get_distances_to_point(x, y, curve))


def get_mean_distances_to_point(x : float, y : float, curve : np.ndarray) -> np.float64:
    """
    Calculates mean distance from given point to each point of curve.
    :param x: x-coordinate of given point
    :param y: y-coordinate of given point
    :param curve: plane curve
    :return mean distance from curve to point.
    """
    if is_empty_curve(curve):
        return 0.0
    return get_sum_distances_to_point(x, y, curve) / get_curve_size(curve)


def homothety_transform(curve : np.ndarray, x : float, y : float, alpha : float) -> np.ndarray:
    """
    Applies homothety transformation with center (x,y) and coefficient alpha to given curve.
    :param curve: plane curve
    :param x: x-coordinate of homothety center
    :param y: y-coordinate of homothety center
    :param alpha: amplifying coefficient
    :return curve transformed.
    """
    return np.add(np.multiply(np.subtract(curve, ([x], [y])), alpha), ([x], [y]))


def get_curve_length(curve : np.ndarray) -> float:
    """
    Curve length calculation
    :param curve: plane curve
    :return length of curve.
    """
    if is_empty_curve(curve):
        return 0.0
    l = np.sum(np.hypot(np.subtract(curve[0][1:], curve[0][:-1]), np.subtract(curve[1][1:], curve[1][:-1])))
    return l + math.hypot(curve[0][0] - curve[0][-1], curve[1][0] - curve[1][-1])


def get_curve_steps(curve : np.ndarray) -> np.ndarray:
    """
    Returns array of curve segments lengths.
    :param curve: plane curve.
    :return array of curve segments lengths.
    """
    return np.append(np.hypot(np.subtract(curve[0][1:], curve[0][:-1]), np.subtract(curve[1][1:], curve[1][:-1])),
                     math.hypot(curve[0][0] - curve[0][-1], curve[1][0] - curve[1][-1]))


def get_curve_length_list(curve : np.ndarray) -> np.ndarray:
    """
    Returns list of curve segments cumulative lengths.
    The last element of output array corresponds to the length of whole curve.
    :param curve: plane curve
    :return array of lengths.
    """
    return np.cumsum(np.append([0.0], get_curve_steps(curve)))


def get_curve_length_from_list(curve_length_list: np.ndarray) -> float:
    """
    Retrieves curve length from list of cumulative lengths.
    :param curve_length_list: list of cumulative lengths.
    :return curve length.
    """
    if len(curve_length_list) == 0:
        return 0.0
    return curve_length_list[-1]


def get_curve_steps_from_list(curve_length_list : np.ndarray) -> np.ndarray:
    """
    Creates list of curve segment lengths from list of cumulative lengths.
    :param curve_length_list: list of cumulative lengths
    :return: list of curve segment lengths.
    """
    return np.subtract(curve_length_list[1:], curve_length_list[:-1])


def get_part_curve_length_from_list(curve_length_list : np.ndarray, i : int, j : int) -> float:
    """
    Returns list of curve between two points given by there indexes.
    :param curve_length_list: list of cumulative lengths
    :param i: index of first point.
    :param j: index of second point.
    :return length of curve part.
    """
    n = len(curve_length_list)
    if n <= 2:
        return 0
    if i == j:
        return 0
    if i >= n:
        return 0

    if j < i:
        print("get_part_curve_length len=", n, " i=", i, " j=", j)
        return curve_length_list[-1] - curve_length_list[i] + curve_length_list[j % (n - 1)] - curve_length_list[0]
    return curve_length_list[j] - curve_length_list[i]


# groups is a list of tuples that contain i1 and i2 - first and last index of point in curve
def get_excl_curve_length_from_list(curve_length_list, groups):
    """

    :param curve_length_list: 
    :param groups: 

    """
    l = get_curve_length_from_list(curve_length_list)
    if len(groups) == 0:
        return l

    s = 0
    for group in groups:
        s += get_part_curve_length_from_list(curve_length_list, group[0], group[1])
    return l - s


def get_part_curve_length(curve, i, j):
    """

    :param curve: 
    :param i: 
    :param j: 

    """
    if is_empty_curve(curve):
        return 0
    n = get_curve_size(curve)
    if i == j:
        return 0

    if j < i:
        print("get_part_curve_length len=", n, " i=", i, " j=", j)

    l = 0
    x1 = curve[0][i % n]
    y1 = curve[1][i % n]
    for k in range(i + 1, j + 1):
        x2 = curve[0][k % n]
        y2 = curve[1][k % n]
        l += math.hypot(x2 - x1, y2 - y1)
        x1 = x2
        y1 = y2
    return l


def get_excl_curve_length(curve, groups):
    """

    :param curve: 
    :param groups: 

    """
    if is_empty_curve(curve):
        return 0

    m = len(groups)
    l = get_curve_length(curve)
    if m == 0:
        return l

    s = 0
    for i in range(m):
        s += get_part_curve_length(curve, groups[i][0], groups[i][1])
    return l - s

# returns square of convex figure dividing it with triangles
def get_convex_curve_square(curve: np.ndarray) -> float:
    """
    Calculates square of convex curve.
    :param curve: convex plane curve
    :return square of interior of given curve.
    """
    [cx, cy] = get_curve_center(curve)

    d = get_distances_to_point(cx, cy, curve)
    s = get_curve_steps(curve)
    p = np.multiply(np.add(np.add(d, np.roll(d, -1)), s), 0.5)
    return np.sum(np.sqrt(np.multiply(np.multiply(p, np.subtract(p, d)), np.multiply(np.subtract(p, s), np.subtract(p, np.roll(d, -1))))))

def get_curvature_over_curve(curve : np.ndarray, curvature : np.ndarray) -> float:
    """
    Calculates integral sum of curvature along curve.
    :param curve: plane curve.
    :param curvature: curvature array (curvature values for each curve point).
    :return integral sum of curvature along curve.
    """
    if is_empty_curve(curve):
        return 0.0
    return np.sum(np.multiply(get_curve_steps(curve), curvature))


def get_horizontal_amplitude(curve : np.ndarray) -> float:
    """
    Calculates horizontal amplitude of curve (max-min along x-axis).
    :param curve: plane curve.
    :return horizontal amplitude of curve.
    """
    if is_empty_curve(curve):
        return 0.0
    return np.max(curve, axis=1)[0] - np.min(curve, axis=1)[0]


def get_vertical_amplitude(curve: np.ndarray) -> float:
    """
    Calculates vertical amplitude of curve (max-min along x-axis).
    :param curve: plane curve.
    :return vertical amplitude of curve.
    """
    if is_empty_curve(curve):
        return 0.0
    return np.max(curve, axis=1)[1] - np.min(curve, axis=1)[1]

def get_curve_amplitude2(curve : np.ndarray) -> [float, float] :
    """
    Calculates horizontal and vertical amplitudes of curve.
    :param curve: plane curve.
    :return list of amplitudes
    """
    return [get_horizontal_amplitude(curve), get_vertical_amplitude(curve)]

def get_curve_amplitude(curve : np.ndarray) -> float :
    """
    Calculates amplitude (linear size) of curve.
    Calculates longest regression line, approximating curve.
    :param curve: plane curve.
    :return list of amplitudes
    """
    return [get_horizontal_amplitude(curve), get_vertical_amplitude(curve)]


def get_curve_linear_size(curve):
    """
    Returns linear size of curve. Uses linear regression to get straight line that
    approximates points of curve. Then finds points where regression line intersects
    curve and returns length of segment. It could be good and not 'computing heavy'
    approximation of curve diameter.
    :param curve: plane curve.
    :return linear size of curve or -1 for failure.
    """

    convex_curve = get_curve_convex_hull(curve)

    ampl_x = get_horizontal_amplitude(convex_curve)
    ampl_y = get_vertical_amplitude(convex_curve)

    reversed = False
    if ampl_y > ampl_x:
        x = convex_curve[1]
        y = convex_curve[0]
        reversed = True
    else:
        x = convex_curve[0]
        y = convex_curve[1]

    z = np.polyfit(x, y, 1)
    polynom = np.poly1d(z)

    points = []
    v = np.subtract(y, polynom(x))
    for i in range(1, len(v)):
        if v[i - 1] * v[i] <= 0.0:
            points.append([(x[i - 1] + x[i]) * 0.5, (y[i - 1] + y[i]) * 0.5])

    if len(points) == 0:
        return -1.0

    if len(points) == 1:
        points.append([x[-1], y[-1]])

    segment_length = math.hypot(points[0][1] - points[1][1], points[0][0] - points[1][0])
    return segment_length


def get_curve_diameter(curve : np.ndarray) -> float :
    """
    Returns max. distance between two points of curve.
    :param curve: plane curve.
    :return max.distance between two points of curve.
    """
    # for curves with a big number of points calculation of diameter
    # like max(distance(Pi, Pj)) for all pairs i < j could be
    # very heavy, so apply this formula for convex hull of given curve.
    # Convex hull will significantly reduce number of points and
    # keep max distance.
    convex_curve = get_curve_convex_hull(curve)
    max_distance = 0.0
    n = get_curve_size(convex_curve)
    for i in range(n):
        x = convex_curve[0][i]
        y = convex_curve[1][i]
        for j in range(i+1, n):
            d = math.hypot(x - convex_curve[0][j], y - convex_curve[1][j])
            max_distance = max(d, max_distance)
    return max_distance


def get_curve_convex_hull(curve : np.ndarray) -> np.ndarray:
    """
    Returns convex hull of curve points.
    :param curve: plane curve.
    :return:  convex figure made from given curve.
    """
    if get_curve_size(curve) > 0:
        hull = ConvexHull(np.permute_dims(curve))
        return np.take(curve, hull.vertices, axis=1)
    return curve


def is_circle(curve : np.ndarray) -> bool :
    """
    Checks if the curve is circle.
    :param curve: plane curve.
    :return:  True if curve is similar to circle, False otherwise.
    """
    if is_empty_curve(curve):
        return False
    [cx, cy] = get_curve_center(curve)
    distances = get_distances_to_point(cx, cy, curve)
    radius_estimated = list_ops.median_value(distances)
    s = get_curve_length(curve)
    radius = s / (2.0 * np.pi)
    # relative tolerance threshold 0.5%
    threshold = 0.5
    #print("p=", 100.0*(math.fabs(radius_estimated - radius)/radius_estimated))
    return 100.0 * (math.fabs(radius_estimated - radius) / radius_estimated) <= threshold


def resample_by_lsq(curve : np.ndarray, w : int=7, po : int=2, n : int=-1) -> np.ndarray:
    """
    Curve is resampled using arclength parameter.
    At every new sample curve point is evaluated using LSQ taking neighbor points.
    This function is used when large variance appears in curve segments length:
    Some segments are shor and some are long.
    Segments here - straight segments between two consecutive points of curve.
    :param curve: plane curve.
    :param w:  window (neighborhood) size (Default value = 7)
    :param po: polynomial order used for approximation (Default value = 2)
    :param n: number of points in output curve (Default value = -1)
    :return curve
    """
    if is_empty_curve(curve):
        return get_empty_curve()
    nc = get_curve_size(curve)
    m = n if n != -1 else nc
    x_vec = []
    y_vec = []
    s = get_curve_length(curve)
    ds = s / m
    w2 = w // 2
    sp = get_curve_length_list(curve)

    i = 0
    for k in range(0, m):
        sk = ds * k
        if k == 0:
            i = 0
        else:
            while i > 0 and i < nc and not (sk <= sp[i] and sk > sp[i - 1]):
                i += 1
            if (i == nc) and (sk > sp[nc - 1] and sk <= s):
                i = nc
        t = []
        x = []
        y = []
        s0 = 0
        for j in range(i - w2, i + w2 + 1):
            l = j if j < 0 else j % nc
            if (j > i - w2 and s0 <= 0) and (t[j - i + w2 - 1] > sp[l]):
                s0 = s
            t.append(sp[l] + s0)
            x.append(curve[0][l])
            y.append(curve[1][l])
        if sk < t[0]:
            sk += s0
        p1 = poly.polyfit(t, x, po)
        x1 = poly.polyval(sk, p1)
        p2 = poly.polyfit(t, y, po)
        y1 = poly.polyval(sk, p2)
        x_vec.append(x1)
        y_vec.append(y1)
        if i == 0:
            i = 1
    return np.array([x_vec, y_vec])


def resample_by_interpolation(curve : np.ndarray, n : int=-1) -> np.ndarray:
    """
    Curve is resampled using arclength parameter (s).
    To get point (x,y) value for given parameter value s interpolation is used.
    :param curve: plane curve
    :param n: number of points in output curve (Default value = -1)
            if n == -1 number of points for output curve doesn't change.
    :return curve: output curve.
    """
    s = get_curve_length(curve)
    nc = get_curve_size(curve)
    m = n if n > 0 else nc
    # current discretization
    length_list = get_curve_length_list(curve)
    fx = interpolate.interp1d(length_list, np.append(curve, [[curve[0][0]], [curve[1][0]]], axis=1)[0], 'cubic')
    fy = interpolate.interp1d(length_list, np.append(curve, [[curve[0][0]], [curve[1][0]]], axis=1)[1], 'cubic')
    # uniform discretization
    t = np.linspace(0.0, s, m, endpoint=False)
    return np.array([fx(t), fy(t)])
