import math
import numpy as np
from scipy import signal
import geometry as geom

#pylint: disable=line-too-long

def get_ellipse(num_points: int, radiusx : float, radiusy : float, orientation : bool = True) -> np.ndarray :
    """
    Returns ellipse curve with given parameters.
    :param num_points: number of points in ellipse.
    :param radiusx: horizontal radius of ellipse.
    :param radiusy: vertical radius of ellipse.
    :param orientation: if True - anticlockwise.
    :return ellipse curve.
    """
    a = 0.0 if orientation else 2.0*np.pi
    b = 2.0*np.pi if orientation else 0.0
    t = np.linspace(a, b, num_points, endpoint=False)
    return np.array([radiusx*np.cos(t), radiusy*np.sin(t)])

def get_circle(num_points : int, radius : float, orientation : bool = True) -> np.ndarray :
    """
    Returns circle curve with given parameters.
    :param num_points: number of points in circle.
    :param radius: radius of circle.
    :param orientation: if True - anticlockwise.
    :return circle curve.
    """
    return get_ellipse(num_points, radius, radius, orientation)

def get_paperclip(num_points : int, radius : float) -> np.ndarray :
    """
    Returns paperclip shaped curve with given parameters.
    :param num_points: number of points in curve.
    :param radius: radius of circle that is part of paperclip curve.
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
    return geom.reverse_curve(np.append(np.append(np.append(z1, z2, axis=1), z3, axis=1), z4, axis=1))

def get_rectangle(num_points : int, a : float, b : float) -> np.ndarray :
    """
    Returns rectangle shaped curve with given parameters.
    :param num_points: number of points in curve.
    :param a: horizontal side of rectangle.
    :param b: vertical side of rectangle
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
    return np.append(np.append(np.append(p1, p2, axis=1), p3, axis=1), p4, axis=1)

def get_lissajous(num_points : int, amplx : float, amply : float, freqx: float, freqy : float, phi : float=0.0) -> np.ndarray :
    """
    Returns Lissajous curve with given parameters.
    x(t)=amplx*sin(freqx*t+phi)     (1)
    y(t)=amply*sin(freqy*t)         (2)
    :param num_points: number of points in curve.
    :param amplx: horizontal amplitude.
    :param amply: vertical amplitude.
    :param freqx: frequency for horizontal component.
    :param freqy: frequency for vertical component.
    :param phi: phase for horizontal component. See equation (1).
    :return Lissajous curve.
    """
    t = np.linspace(0.0, 2.0*np.pi, num_points, endpoint=False)
    return np.array([amplx*np.sin(freqx*t + phi), amply*np.sin(freqy*t)])

def get_figure_8(num_points : int, ampl : float) -> np.ndarray :
    """
    Returns '8'shape curve Lissajous curve with given parameters.
    x(t)=ampl*sin(2*t)     (1)
    y(t)=ampl*sin(t)       (2)
    :param num_points: number of points in curve.
    :param ampl: amplitude.
    :return '8' shaped curve.
    """
    return get_lissajous(num_points, ampl, ampl, 2.0, 1.0)


def get_exotic_curve_1(num_points : int) -> np.ndarray :

    h = 90.0
    R = 70.0
    r = R / 3.0

    L = 6.0*h + 3*np.pi*r + np.pi*R
    n1 = int((h * num_points) / L)
    x1 = np.full(n1, R)
    y1 = np.linspace(0.0, h, n1, endpoint=False)

    n2 = int((np.pi * r * num_points) / L)
    t2 = np.linspace(0.0, np.pi, n2, endpoint=False)
    x2 = r * np.cos(t2) + R - r
    y2 = r * np.sin(t2) + h

    n3 = int(((h + h)*num_points)/L)
    x3 = np.full(n3, R - 2.0 * r)
    y3 = np.linspace(h, -h, n3, endpoint=False)

    x4 = r * np.cos(t2)
    y4 = -r * np.sin(t2) - h

    x5 = np.full(n3, -R + 2.0 * r)
    y5 = np.linspace(-h, h, n3, endpoint=False)

    x6 = r * np.cos(t2) - R + r
    y6 = r * np.sin(t2) + h

    x7 = np.full(n1, -R)
    y7 = np.linspace(h, 0.0, n1, endpoint=False)

    x8 = -R * np.cos(t2)
    y8 = -R * np.sin(t2)

    x = np.append(x1, np.append(x2, np.append(x3, np.append(x4, np.append(x5, np.append(x6, np.append(x7, x8)))))))
    y = np.append(y1, np.append(y2, np.append(y3, np.append(y4, np.append(y5, np.append(y6, np.append(y7, y8)))))))
    return np.array([x, -y])
    

def get_exotic_curve_2(num_points : int) -> np.ndarray :

    h = 100.0
    r = 20.0

    L = 4.0*h + 4.0*np.pi*r
    n1 = int((h * num_points) / L)
    x1 = np.full(n1, r)
    y1 = np.linspace(0.0, h, n1, endpoint=False)

    n2 = int((np.pi * r * num_points) / L)
    t2 = np.linspace(0.0, np.pi, n2, endpoint=False)
    x2 = r * np.cos(t2)
    y2 = r * np.sin(t2) + h

    n3 = int((h * num_points) / L)
    x3 = np.full(n3, -r)
    y3 = np.linspace(h, 0, n3, endpoint=False)

    n4 = int((3.0 * np.pi * r * num_points) / L)
    t4 = np.linspace(np.pi, 2.0 * np.pi + np.pi * 0.5 , n4, endpoint=False)
    x4 = r * np.cos(t4)
    y4 = r * np.sin(t4)

    y5 = np.full(n1, r)
    x5 = np.linspace(0.0, -h, n1, endpoint=False)

    t6 = np.linspace(np.pi * 0.5, 3.0 * np.pi * 0.5, n2, endpoint=False)
    x6 = r * np.cos(t6) - h
    y6 = r * np.sin(t6)

    y7 = np.full(n1, -r)
    x7 = np.linspace(-h, 0.0, n1, endpoint=False)

    n8 = int((0.25 * np.pi * r * num_points) / L)
    t8 = np.linspace(3.0 * np.pi * 0.5, 2.0 * np.pi, n8, endpoint=False)
    x8 = r * np.cos(t8)
    y8 = r * np.sin(t8)

    x = np.append(x1, np.append(x2, np.append(x3, np.append(x4, np.append(x5, np.append(x6, np.append(x7, x8)))))))
    y = np.append(y1, np.append(y2, np.append(y3, np.append(y4, np.append(y5, np.append(y6, np.append(y7, y8)))))))
    return np.array([x, y])


def get_touching_circles(num_points : int, radiuses : np.ndarray) -> np.ndarray :
    """
    Returns curve that contains circles with given radiuses that has one common touching point.
    If radius is positive then circle has positive (anticlockwise) orientation.
    If radius is negative then circle has clockwise orientation and absolute value is taken for circle radius.
    :param num_points: number of points in curve.
    :param radiuses: list of radiuses (could be negative).
    :return curve made from touching circles.
    """
    if not radiuses:
        return geom.get_empty_curve()
    
    t1 = np.linspace(0.0, 2.0*np.pi, num_points, endpoint=False)
    t2 = np.linspace(np.pi, -np.pi, num_points, endpoint=False)
    x = np.array([])
    y = np.array([])
    for i in range(len(radiuses)):
        r = radiuses[i]
        x = np.append(x, math.fabs(r)*np.cos(t1 if r > 0.0 else t2) - r)
        y = np.append(y, math.fabs(r)*np.sin(t1 if r > 0.0 else t2))
    return np.array([x, y])

def get_touching_8(num_points : int, radius : float) -> np.ndarray :
    """
    Returns curve that contains two touching circles of different orientation (shape like oo).
    :param num_points: number of points in curve.
    :param radius: radius of each circle.
    :return 'oo'-shape curve.
    """
    return get_touching_circles(num_points, [radius, -radius])
