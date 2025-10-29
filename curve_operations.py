import numpy as np

""""
curve is always represented by np.array([x1,x2,...], [y1,y2,...]])
curve.shape = (2,n)
"""

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
    #print("First t[0]=", t[0], " last =", t[-1], " 2pi=", 2.0*np.pi)
    #print(np.array([radiusx*np.cos(t), radiusy*np.sin(t)]))
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
    #print("First sin=", curve[0][0], curve[1][0], " last =", curve[0][-1], curve[1][-1])
    z2 = np.array([np.linspace(0.0, 3.0*radius, n2 // 2, endpoint=False), np.full(n2 // 2, radius)])
    #print("First z1 = ", z1[0][0], z1[1][0], " last z1=", z1[0][-1], z1[1][-1])
    t3 = np.linspace(2.5 * np.pi, 1.5 * np.pi, n1 - ( n1 // 2), endpoint=False)

    z3 = np.array([3.0 * radius + radius * np.cos(t3), radius * np.sin(t3)])

    #print("First z2 = ", z2[0][0], z2[1][0], " last z2=", z2[0][-1], z2[1][-1])
    z4 = np.array([np.linspace(3.0*radius, 0.0, n2 - (n2 // 2), endpoint=False), np.full(n2 - (n2 // 2), -radius)])
    #print("First z3 = ", z3[0][0], z3[1][0], " last z3=", z3[0][-1], z3[1][-1])
    #curve = np.append(np.append(np.append(curve, z1, axis=1), z2, axis=1), z3, axis=1)
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