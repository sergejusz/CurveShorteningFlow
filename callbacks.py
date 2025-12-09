import os
import cv2
import numpy as np
import geometry as geom
import image_operations as img_ops
import CallbackData as cbd


# pylint: disable=line-too-long

def get_curve_image_path(folder_path: str, extent: str, iteration: int, max_iterations: int) -> str:
    m = max(len(str(max_iterations)), 5)
    return os.path.join(folder_path, 'image' + (str(iteration)).zfill(m) + extent)

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
    ampl = min(geom.get_curve_amplitudes(curve))
    if ampl >= 60:
        return 20.0
    if ampl >= 40:
        return 12.0
    if ampl >= 20:
        return 0.2*ampl + 2.0
    if ampl >= 10:
        return 0.3*ampl
    return 3.0

def estimate_curve_diameter(curve: np.ndarray, is_circle_shape: bool) -> float:
    """
    Returns diameter of curve.
    :param curve: curve data.
    :param is_circle_shape: True if curve is circle.
    :return diameter of curve.
    """
    # is_circle_shape is not used for now
    return geom.get_curve_linear_size(curve)

def detect_last_call(curve: np.ndarray, iteration: int, is_circle_shape: bool, max_iterations: int, minimal_diameter: float) -> bool:
    last_iteration = max_iterations > 0 and iteration == max_iterations
    size_too_small = False
    object_size = minimal_diameter + 100.0
    if last_iteration or (max(geom.get_curve_amplitudes(curve)) <= minimal_diameter):
        object_size = estimate_curve_diameter(curve, is_circle_shape)
        size_too_small = object_size <= minimal_diameter

    last_call = size_too_small or last_iteration
    if last_call:
        print("Object linear size=", object_size)
    return last_call

def contour_view_callback(curve : np.ndarray, curvature : np.ndarray, curve_length : float, iteration : int, is_circle_shape: bool, obj: cbd.CallbackData) -> bool:
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
        # recognize if it should be the last call of callback function
        last_call = detect_last_call(curve, iteration, is_circle_shape, obj.max_iterations, obj.diameter)
        # always display curve at the last iteration if it is the last curve in the list
        if (iteration % obj.save_to_file_counter == 0) or (last_call and obj.first_curve):
            file_path = get_curve_image_path(obj.path, '.png', iteration, obj.max_iterations)
            image_exists = os.path.exists(file_path)
            background_color = (0,0,0) if obj.jet_colors else obj.background_color
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists \
                else np.full((obj.rows, obj.cols, 3), obj.background_color, np.uint8)
            if not img is None:
                if obj.jet_colors:
                    img_ops.draw_curve_lines_curvature(img, curve, curvature, obj.line_thickness)
                else:
                    img_ops.draw_curve_lines(img, curve, obj.foreground_color, obj.line_thickness)
                if obj.last_curve:
                    if obj.jet_colors:
                        img = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
                    if obj.gauss_blurring > 0:
                        img = cv2.GaussianBlur(img, (obj.gauss_blurring, obj.gauss_blurring), 0)

                cv2.imwrite(file_path, img)

        # terminate flow if number of iterations is exhausted or curve size is too small
        return last_call
    return True

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
        # recognize if it should be the last call of callback function
        last_call = detect_last_call(curve, iteration, is_circle_shape, obj.max_iterations, obj.diameter)
        # always display curve at the last iteration if it is the last curve in the list
        if (iteration % obj.save_to_file_counter == 0) or (last_call and obj.first_curve):
            file_path = get_curve_image_path(obj.path, '.png', iteration, obj.max_iterations)
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists \
                else np.full((obj.rows, obj.cols, 3), obj.background_color, np.uint8)
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj.foreground_color, obj.line_thickness)
                normal_field = geom.get_normal_field(curve)
                normal_unit_field = geom.normalize(normal_field)
                skip_samples = get_sample_skip_count_for_vector(curve_length, geom.get_curve_size(curve))
                scaling_factor = get_vector_scaling_factor(curve)
                img_ops.display_shortening_field(img, curve, curvature, normal_unit_field,
                    obj.foreground_color, delta_n=skip_samples, magnify=scaling_factor)
                if obj.last_curve and obj.gauss_blurring > 0:
                    img = cv2.GaussianBlur(img, (obj.gauss_blurring, obj.gauss_blurring), 0)
                cv2.imwrite(file_path, img)

        # terminate flow if number of iterations is exhausted or curve size is too small
        return last_call
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
        if iteration % obj.history_skip == 0:
            curves.append(curve)
            if len(curves) > obj.history_length:
                del curves[0]

        # recognize if it should be the last call of callback function
        last_call = detect_last_call(curve, iteration, is_circle_shape, obj.max_iterations, obj.diameter)
        # always display curve at the last iteration if it is the last curve in the list
        if (iteration % obj.save_to_file_counter == 0) or (last_call and obj.first_curve):
            file_path = get_curve_image_path(obj.path, '.png', iteration, obj.max_iterations)
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists \
                else np.full((obj.rows, obj.cols, 3), obj.background_color, np.uint8)

            if not img is None:
                m = len(curves)
                i = 0
                for curve_ in curves:
                    if obj.jet_colors:
                        curvature_ = geom.get_curvature(curve_)
                        img_ops.draw_curve_lines_curvature(img, curve_, curvature_, obj.line_thickness)
                    else:
                        img_ops.draw_curve_lines(img, curve_, obj.history_colors[m - i - 1], obj.line_thickness)
                    i += 1

                if obj.last_curve:
                    if obj.jet_colors:
                        img = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
                if obj.last_curve and obj.gauss_blurring > 0:
                    img = cv2.GaussianBlur(img, (obj.gauss_blurring, obj.gauss_blurring), 0)
                cv2.imwrite(file_path, img)

        # terminate flow if number of iterations is exhausted or curve size is too small
        return last_call
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
        # recognize if it should be the last call of callback function
        last_call = detect_last_call(curve, iteration, is_circle_shape, obj.max_iterations, obj.diameter)
        # always display curve at the last iteration if it is the last curve in the list
        if (iteration % obj.save_to_file_counter == 0) or (last_call and obj.first_curve):
            file_path = get_curve_image_path(obj.path, '.png', iteration, obj.max_iterations)
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.zeros((obj.rows, obj.cols, 3), np.uint8)
            cv2.floodFill(img, None, (1, 1), obj.background_color)
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj.foreground_color)
                if min(curvature) >= 0:
                    img_ops.fill_convex_curve(img, curve, obj.foreground_color)
                else:
                    img_ops.fill_curve(img, curve, curvature, obj.foreground_color)
                if obj.last_curve and obj.gauss_blurring > 0:
                    img = cv2.GaussianBlur(img, (obj.gauss_blurring, obj.gauss_blurring), 0)
                cv2.imwrite(file_path, img)

        # terminate flow if number of iterations is exhausted or curve size is too small
        return last_call
    return True
