import math
from pytest import approx
from pytest import approx
import numpy as np
import geometry as geom
import curves


def test_get_curvature():
    radius1 = 10.0
    curvature1 = geom.get_curvature(curves.get_circle(100, radius1))
    assert math.fabs(np.max(curvature1) - np.min(curvature1)) == approx(0.0, abs = 0.1)
    assert np.mean(curvature1) == approx(1.0 / radius1, rel = 0.01)
    assert np.median(curvature1) == approx(1.0 / radius1, rel = 0.01)

    curvature2 = geom.get_curvature(curves.get_circle(200, radius1))
    assert math.fabs(np.max(curvature2) - np.min(curvature2)) == approx(0.0, abs = 0.05)
    assert np.mean(curvature2) == approx(1.0 / radius1, rel = 0.001)
    assert np.median(curvature2) == approx(1.0 / radius1, rel = 0.001)

    radius2 = 20.0
    curvature3 = geom.get_curvature(curves.get_circle(1000, radius2))
    assert math.fabs(np.max(curvature3) - np.min(curvature3)) == approx(0.0, abs = 0.05)
    assert np.mean(curvature3) == approx(1.0 / radius2, rel = 0.001)
    assert np.median(curvature3) == approx(1.0 / radius2, rel = 0.001)

    curvature4 = geom.get_curvature(curves.get_circle(1000, radius2), w=7, po=2)
    assert math.fabs(np.max(curvature4) - np.min(curvature4)) == approx(0.0, abs = 0.02)
    assert np.mean(curvature4) == approx(1.0 / radius2, rel = 0.0001)
    assert np.median(curvature4) == approx(1.0 / radius2, rel = 0.0001)


def test_get_curve_steps():
    curve = np.array([[0, 1, 2, 3, 3, 3, 3, 2, 1, 0, 0, 0], [0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 2, 1]])
    assert curve.shape[1:][0] == 12
    s = geom.get_curve_steps(curve)
    print(s)
    assert len(s) == 12
    expected = np.ones(12)
    assert len(expected) == 12
    for i in range(len(expected)):
        assert expected[i] == s[i]


def test_get_curve_length():
    assert geom.get_curve_length(np.array([[0, 1, 1], [0, 0, 1]])) == approx(2.0 + math.sqrt(2.0))
    assert geom.get_curve_length(np.array([[0,1,2,3,3,3,3,2,1,0,0,0],[0,0,0,0,1,2,3,3,3,3,2,1]])) == 12.0

def test_get_convex_curve_square():
    assert geom.get_convex_curve_square(np.array([[0,1,1,0],[0,0,1,1]])) == approx(1.0)
    assert geom.get_convex_curve_square(np.array([[0, 2, 1], [0, 0, 3]])) == approx(3.0)
    assert geom.get_convex_curve_square(np.array([[0, 1, 1, 0], [0, 0, 10, 10]])) == approx(10.0)
    radius1 = 10.0
    assert geom.get_convex_curve_square(curves.get_circle(200, radius1)) == approx(np.pi * radius1 * radius1, rel=0.001)

def test_rotate_curve():
    #rectangle
    assert geom.rotate_curve(np.array([[2, -2, -2, 2],[1, 1, -1, -1]]), np.pi * 0.5) == approx(np.array([[1, 1, -1, -1],[-2, 2, 2, -2]]))
    assert geom.rotate_curve(np.array([[3, -1, -1, 3],[2, 2, 0, 0]]), np.pi * 0.5) == approx(np.array([[2, 2, 0, 0],[-1, 3, 3, -1]]))

def test_get_curve_diameter():
    assert approx(5.0) == geom.get_curve_diameter(np.array([[0, 1, 1, 2, 2, 3, 3, 2, 2, 1, 1, 0], [0, 0, 1, 1, 2, 2, 4, 4, 3, 3, 2, 2]]))
    assert approx(math.sqrt(29.0)) == geom.get_curve_diameter(np.array([[0, 1, 1, 5, 5, 0], [0, 0, 1, 1, 2, 2]]))

def test_rotation_number():
    circle = curves.get_circle(100, 10.0)
    assert 1 == geom.get_rotation_number(circle)
    assert -1 == geom.get_rotation_number(geom.reverse_curve(circle))
    assert 1 == geom.get_rotation_number(geom.translate(geom.homothety_transform(circle, -2.0, 2.0, 2.5), 2.0, 5.0))
    assert -1 == geom.get_rotation_number(geom.reverse_curve(geom.translate(geom.homothety_transform(circle, -2.0, 2.0, 2.5), 2.0, 5.0)))
    assert 1 == geom.get_rotation_number(curves.get_paperclip(200, 10.0))
    assert -1 == geom.get_rotation_number(geom.reverse_curve(curves.get_paperclip(200, 10.0)))
    # '8' shape curve
    assert 0 == geom.get_rotation_number(curves.get_lissajous(200, 10.0, 10.0, 2.0, 1.0))
    assert 0 == geom.get_rotation_number(curves.get_lissajous(200, 10.0, 10.0, 1.0, 2.0))
    assert 1 == geom.get_rotation_number(curves.get_touching_circles(200, [10.0, -10.0, 20.0]))
    assert 3 == geom.get_rotation_number(curves.get_touching_circles(200, [10.0, 20.0, 30.0]))
    assert 5 == geom.get_rotation_number(curves.get_touching_circles(200, [10.0, 20.0, 30.0, 25.0, 15.0]))
    assert 0 == geom.get_rotation_number(curves.get_touching_circles(200, [10.0, 20.0, 30.0, -15.0, -25.0, -5.0]))
