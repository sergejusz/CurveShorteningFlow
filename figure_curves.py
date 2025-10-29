import curve_operations as curve_ops


def get_circle_curve(arg_radius, arg_num_points):
    """
    Returns circle curve with given radius and number of points or empty curve.
    :param arg_radius:circle radius.
    :param arg_num_points:  number of points for curve specified by user.
    If not specified then number of points is calculated using radius.
    :return circle curve or empty curve.
    """
    if arg_radius == 0:
        print("ERROR: radius for circle cannot be zero")
        return curve_ops.get_empty_curve()

    num_points = arg_radius * 4 if arg_num_points == 0 else arg_num_points
    return curve_ops.get_circle(0, 0, arg_radius, num_points)

def get_ellipse_curve(arg_radius_x, arg_radius_y, arg_num_points):
    """
    Returns ellipse curve with given radiuses and number of points or empty curve.
    :param arg_radius_x: horizontal radius.
    :param arg_radius_y: vertical radius.
    :param arg_num_points:  number of points for curve specified by user.
    If not specified then number of points is calculated using radiuses.
    :return ellipse curve or empty curve.
    """
    if arg_radius_x == 0 or arg_radius_y == 0:
        print("ERROR: radiuses for ellipse cannot be zero")
        return curve_ops.get_empty_curve()

    if arg_num_points == 0:
        num_points = (arg_radius_x + arg_radius_y) * 2
    else:
        num_points = arg_num_points
    return curve_ops.get_ellipse(0, 0, arg_radius_x, arg_radius_y, num_points)

def get_paperclip_curve(arg_radius, arg_num_points):
    """
    Returns paperclip curve with given radius and number of points or empty curve.
    :param arg_radius: radius.
    :param arg_num_points:  number of points for curve specified by user.
    If not specified then number of points is calculated using radius.
    :return paperclip curve or empty curve.
    """
    if arg_radius == 0:
        print("ERROR: radius for paperclip cannot be zero")
        return curve_ops.get_empty_curve()

    num_points = arg_radius * 8 if arg_num_points == 0 else arg_num_points
    return curve_ops.get_paperclip(0, 0, arg_radius, num_points)


def get_rectangle_curve(arg_side_x, arg_side_y, arg_num_points):
    """
    Returns rectangle curve with given sides and number of points or empty curve.
    :param arg_side_x: horizontal side.
    :param arg_side_y: vertical side.
    :param arg_num_points:  number of points for curve specified by user.
    If not specified then number of points is calculated using radius.
    :return rectangular curve or empty curve.
    """
    if arg_side_x == 0 or arg_side_y == 0:
        print("ERROR: rectangle sides cannot be zero")
        return curve_ops.get_empty_curve()

    num_points = (arg_side_x + arg_side_y) * 2 if arg_num_points == 0 else arg_num_points
    return curve_ops.get_rectangle(0, 0, arg_side_x, arg_side_y, num_points)
