import math
from pytest import approx
from pytest import approx
import numpy as np
import geometry as geom
import curve_operations as curve_ops


def test_get_curvature():
    radius1 = 10.0
    curvature1 = geom.get_curvature(curve_ops.get_circle(0.0, 0.0, radius1, 100))
    assert math.fabs(np.max(curvature1) - np.min(curvature1)) == approx(0.0, abs = 0.1)
    assert np.mean(curvature1) == approx(1.0 / radius1, rel = 0.01)
    assert np.median(curvature1) == approx(1.0 / radius1, rel = 0.01)

    curvature2 = geom.get_curvature(curve_ops.get_circle(0.0, 0.0, radius1, 200))
    assert math.fabs(np.max(curvature2) - np.min(curvature2)) == approx(0.0, abs = 0.05)
    assert np.mean(curvature2) == approx(1.0 / radius1, rel = 0.001)
    assert np.median(curvature2) == approx(1.0 / radius1, rel = 0.001)

    radius2 = 20.0
    curvature3 = geom.get_curvature(curve_ops.get_circle(0.0, 0.0, radius2, 1000))
    assert math.fabs(np.max(curvature3) - np.min(curvature3)) == approx(0.0, abs = 0.05)
    assert np.mean(curvature3) == approx(1.0 / radius2, rel = 0.001)
    assert np.median(curvature3) == approx(1.0 / radius2, rel = 0.001)

    curvature4 = geom.get_curvature(curve_ops.get_circle(0.0, 0.0, radius2, 1000), w=7, po=2)
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
    assert geom.get_convex_curve_square(curve_ops.get_circle(0.0, 0.0, radius1, 200)) == approx(np.pi * radius1 * radius1, rel=0.001)