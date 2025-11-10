import os
import cv2
import numpy as np
from enums import CallbackArgs
from enums import HistoryViewStyle
import geometry as geom
import image_operations as img_ops


# pylint: disable=line-too-long


def get_sample_skip_count_for_vector(curve_length : float, num_samples : int) -> int :
    """
    Returns number of samples to skip when drawing normal vectors.
    :param curve_length: curve length.
    :param num_samples: number of samples of curve.
    """
    ds = curve_length / num_samples
    if ds < 0.1:
        return 30
    if ds < 0.2:
        return 20
    if ds < 0.5:
        return 10
    if ds < 1.0:
        return 5
    if ds < 2.0:
        return 2
    return 1

def get_vector_scaling_factor(curve : np.ndarray) -> float:
    """
    Calculates scaling factor to display normal vectors.
    :param curve: curve points.
    :return scaling_factor.
    """
    ampl = min(geom.get_curve_amplitude(curve))
    if ampl >= 60:
        return 20.0
    if ampl >= 40:
        return 12.0
    if ampl >= 20:
        return 0.2*ampl + 2.0
    if ampl >= 10:
        return 0.3*ampl
    return 3.0

def estimate_curve_diameter(curve: np.ndarray, curve_length: float, is_circle_shape: bool) -> float:
    """
    Returns diameter of curve.
    :param curve: curve data.
    :param curve_length: curve length.
    :param is_circle_shape: True if curve is circle.
    :return diameter of curve.
    """
    # is_circle_shape is not used for now
    return geom.get_curve_linear_size(curve)

# callback function returns True to terminate flow, False otherwise
def vector_view_callback(curve, curvature, curve_length, iteration, is_circle_shape, obj):
    """
    Function is called at each iteration by CurveShortener class object.
    Visualizes current curve together with vectors showing
    direction of curve shortening flow.
    :param curve: curve data.
    :param curvature: curvature array.
    :param curve_length: length of curve.
    :param iteration: number of current iteration.
    :param is_circle_shape: True if curve is circle.
    :param obj: object that contains environmental data.
    """
    print("iter=", iteration, " curve arc length=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[CallbackArgs.SAVETOFILECOUNTER]
        if iteration % n == 0:
            rows = obj[CallbackArgs.ROWS]
            cols = obj[CallbackArgs.COLS]
            file_path = os.path.join(obj[CallbackArgs.PATH],
                'image' + (str(iteration)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists \
                else np.full((rows, cols, 3), obj[CallbackArgs.BACKGROUNDCOLOR], np.uint8)
            if not img is None:
                fg_color = obj[CallbackArgs.FOREGROUNDCOLOR]
                img_ops.draw_curve_lines(img, curve, fg_color)
                normal_field = geom.get_normal_field(curve)
                normal_unit_field = geom.normalize(normal_field)
                skip_samples = get_sample_skip_count_for_vector(curve_length, geom.get_curve_size(curve))
                scaling_factor = get_vector_scaling_factor(curve)
                img_ops.display_shortening_field(img, curve, curvature, normal_unit_field,
                    fg_color, delta_n=skip_samples, magnify=scaling_factor)
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MAXITERATIONS]
        minimal_diameter = obj[CallbackArgs.DIAMETER]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return ((max_iterations > 0 and iteration == max_iterations) or
                estimate_curve_diameter(curve, curve_length, is_circle_shape) < minimal_diameter)
    return True


def contour_view_callback(curve, curvature, curve_length, iteration, is_circle_shape, obj):
    """
    Function is called at each iteration by CurveShortener class object.
    Visualizes current curve with specified color.
    :param curve: curve data.
    :param curvature: curvature array.
    :param curve_length: length of curve.
    :param iteration: number of current iteration.
    :param is_circle_shape: True if curve is circle.
    :param obj: object that contains environmental data.
    """
    print("iter=", iteration, " curve arclength=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[CallbackArgs.SAVETOFILECOUNTER]
        if iteration % n == 0:
            rows = obj[CallbackArgs.ROWS]
            cols = obj[CallbackArgs.COLS]
            file_path = os.path.join(obj[CallbackArgs.PATH], 'image' + (str(iteration)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists \
                else np.full((rows, cols, 3), obj[CallbackArgs.BACKGROUNDCOLOR], np.uint8)
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj[CallbackArgs.FOREGROUNDCOLOR])
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MAXITERATIONS]
        minimal_diameter = obj[CallbackArgs.DIAMETER]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return ((max_iterations > 0 and iteration == max_iterations) or
                estimate_curve_diameter(curve, curve_length, is_circle_shape) < minimal_diameter)
    return True

# pylint: disable=dangerous-default-value
def history_view_callback(curve, curvature, curve_length, iteration, is_circle_shape, obj, curves = []):
    """
    Function is called at each iteration by CurveShortener class object.
    Visualizes current curve together with few previous curves.
    :param curve: curve data.
    :param curvature: curvature array.
    :param curve_length: length of curve.
    :param iteration: number of current iteration.
    :param is_circle_shape: True if curve is circle.
    :param obj: object that contains environmental data.
    """
    print("iter=", iteration, " curve arclength=", geom.get_curve_length(curve))

    if iteration == 0:
        curves.clear()

    if obj is not None:
        if iteration % HistoryViewStyle.SKIPITERATIONS == 0:
            curves.append(curve)
            if len(curves) > HistoryViewStyle.MAXCOUNT:
                del curves[0]

        n = obj[CallbackArgs.SAVETOFILECOUNTER]
        if iteration % n == 0:
            rows = obj[CallbackArgs.ROWS]
            cols = obj[CallbackArgs.COLS]
            file_path = os.path.join(obj[CallbackArgs.PATH], 'image' + (str(iteration)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists \
                else np.full((rows, cols, 3), obj[CallbackArgs.BACKGROUNDCOLOR], np.uint8)

            if not img is None:
                m = len(curves)
                i = 0
                for curve_ in curves:
                    img_ops.draw_curve_lines(img, curve_, obj[CallbackArgs.HISTORYCOLORS][m - i - 1])
                    i += 1
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MAXITERATIONS]
        minimal_diameter = obj[CallbackArgs.DIAMETER]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return ((max_iterations > 0 and iteration == max_iterations) or
                estimate_curve_diameter(curve, curve_length, is_circle_shape) < minimal_diameter)
    return True
# pylint: enable=dangerous-default-value

def solid_view_callback(curve, curvature, curve_length, iteration, is_circle_shape, obj):
    """
    Function is called at each iteration by CurveShortener class object.
    Visualizes current curve filled with some specified color.
    :param curve: curve data.
    :param curvature: curvature array.
    :param curve_length: length of curve.
    :param iteration: number of current iteration.
    :param is_circle_shape: True if curve is circle.
    :param obj: object that contains environmental data.
    """
    print("iter=", iteration, " curve arclength=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[CallbackArgs.SAVETOFILECOUNTER]
        if iteration % n == 0:
            rows = obj[CallbackArgs.ROWS]
            cols = obj[CallbackArgs.COLS]
            file_path = os.path.join(obj[CallbackArgs.PATH], 'image' + (str(iteration)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.zeros((rows, cols, 3), np.uint8)

            cv2.floodFill(img, None, (1, 1), obj[CallbackArgs.BACKGROUNDCOLOR])
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj[CallbackArgs.FOREGROUNDCOLOR])
                if min(curvature) >= 0:
                    img_ops.fill_convex_curve(img, curve, obj[CallbackArgs.FOREGROUNDCOLOR])
                else:
                    img_ops.fill_curve(img, curve, curvature, obj[CallbackArgs.FOREGROUNDCOLOR])
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MAXITERATIONS]
        minimal_diameter = obj[CallbackArgs.DIAMETER]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return ((max_iterations > 0 and iteration == max_iterations) or
                estimate_curve_diameter(curve, curve_length, is_circle_shape) < minimal_diameter)

    return True
