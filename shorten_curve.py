import os
import sys
from enum import IntEnum
import argparse
from functools import cmp_to_key
import cv2
import numpy as np
import geometry as geom
import CurveExtractor as ce
import CurveShortener as cs
import image_operations as img_ops
import color_operations as color_ops
import curve_operations as curve_ops
import figure_curves as fig_curves
import ColorInterpolator as ci
from enums import HistoryViewStyle
import callbacks as cb


# pylint: disable=line-too-long

class ViewStyle(IntEnum):
    """ """
    INVALID = -1
    CONTOUR = 1
    SOLID = 2
    VECTOR = 3
    HISTORY = 4

def parse_command_line():
    """
    :return Command line arguments.
    """
    parser = argparse.ArgumentParser(prog='shorten_curve.py', description='Curve shortening flow demo', epilog='Text at the bottom of help')
    parser.add_argument('image_path', help='source image file containing closed simple curve or CIRCLE or ELLIPSE or PAPERCLIP curve')
    parser.add_argument('dest_folder', help='Folder path to store output images')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=100, help='max number of iterations')
    parser.add_argument('-p', '--preserve_area', required=False, action='store_true', help='preserve area')
    parser.add_argument('-s', '--save_every_n', type=int, required=False, default=5,
        help='How often image with curve is saved. Default is 5 - every 5th image is saved')
    parser.add_argument('--accumulate', type=int, required=False, default=20, help='number of curves in history buffer. Default 20.')
    parser.add_argument('-v', '--view', required=False, default='contour', choices=['contour', 'solid', 'vector', 'history'],
        help='View style of curves under flow')
    parser.add_argument('-m', '--median_filter', type=int, required=False, default=0, choices=[3, 5, 7, 9],
        help='apply median filter for source image with windowsize n')
    parser.add_argument('-n', '--number_curves', type=int, required=False, default=1, choices=[1, 2, 3, 4, 5],
        help='sets number of curves to apply shortening flow')
    parser.add_argument('-c', '--color_palette', type=str, required=False, default='blue',
                        choices=['blue', 'green', 'red'], help='sets coloring for visualization of filled curves')
    parser.add_argument('--num_points', type=int, required=False, default=0, help='number of points for curve')
    parser.add_argument('--save_info', required=False, action='store_true', help='save additional information to files')
    parser.add_argument('--radius', type=int, required=False, default=0, help='circle radius')
    parser.add_argument('--radius_x', type=int, required=False, default=0, help='ellipse radius for x axis')
    parser.add_argument('--radius_y', type=int, required=False, default=0, help='ellipse radius for y axis')
    parser.add_argument('--ascending', required=False, action='store_true', help='process multiple curves from shortest to longest. Default order is from longest to shortest.')
    parser.add_argument('--curve_no', type=int, required=False, default=0, choices={1,2,3,4,5}, help='process only specified curve with given number. Number starts from 1.')
    parser.add_argument('--bg_color', type=str, required=False, default='000000', help='background color for contour, vector and history views')
    parser.add_argument('--fg_color', type=str, required=False, default='ffffff', help='foreground color for contour, vector and history views')
    parser.add_argument('--side_x', type=int, required=False, default=0, help='side length for x axis')
    parser.add_argument('--side_y', type=int, required=False, default=0, help='side length for y axis')
    parser.add_argument('--height', type=int, required=False, default=0, help='image height to display curves')
    parser.add_argument('--width', type=int, required=False, default=0, help='image width to display curves')
    return parser.parse_args()

def get_extension(file_path):
    """
    Returns extension of file path.
    :param file_path: file path to get extension from.
    :return file extension string.
    """
    ext = file_path.rpartition('.')[-1]
    return ext

def accept_extension(extent):
    """
    Returns True if extension is accepted.
    :param extent:
    :return True if extension is for graphics file.
    """
    ext = extent.strip().lower()
    if len(ext) == 0:
        return False
    return len([s for s in ["bmp", "png", "jpg", "jpeg"] if s == ext]) == 1


def get_history_colors(fg_color, bg_color, num_levels):
    """
    Returns list of colors for curves mixing foreground and background colors
    so that color for most recent curve is nearest to foreground color
    and color for oldest curve is nearest to background color.
    :param fg_color: foreground color.
    :param bg_color: background color.
    :param num_levels: number of history curves.
    :return list of colors.
    """
    color_interpolator = ci.ColorInterpolator(fg_color, bg_color, num_levels)
    history_colors = []
    for i in range(num_levels):
        (r, g, b) = color_interpolator.calculate(i)
        history_colors.append((int(r), int(g), int(b)))
    return history_colors


def extract_curve(img):
    """
    Returns curve data extracted from given image.
    :param img: cv2 image containing curve.
    :return curve data.
    """
    curve_extractor = ce.CurveExtractor()
    signal_color = img_ops.get_signal_color()
    curve = curve_extractor.extract(img, signal_color)
    # remove extracted curve from image
    img_ops.clear_curve(img, curve, 5, 5, 0)
    return curve

def get_color_palette(coloring_name):
    """
    Returns color palette for given coloring name.
    :param coloring_name: 
    :return color palette.
    """
    upper_name = coloring_name.upper()
    if upper_name == 'RED':
        return color_ops.get_red_palette()
    if upper_name == 'BLUE':
        return color_ops.get_blue_palette()
    if upper_name == 'GREEN':
        return color_ops.get_green_palette()
    return []


def get_background_color(coloring_name):
    """
    Returns background color for given coloring name.
    :param coloring_name: 
    :return color.
    """
    upper_name = coloring_name.upper()
    if upper_name == 'RED':
        return color_ops.get_red_background_color()
    if upper_name == 'BLUE':
        return color_ops.get_blue_background_color()
    if upper_name == 'GREEN':
        return color_ops.get_green_background_color()
    return ()


def get_view_style(view_style):
    """
    Returns visual style for given string.
    :param view_style: string.
    :return type of visual style.
    """
    upper_name = view_style.upper()
    if upper_name == 'CONTOUR':
        return ViewStyle.CONTOUR
    if upper_name == 'SOLID':
        return ViewStyle.SOLID
    if upper_name == 'VECTOR':
        return ViewStyle.VECTOR
    if upper_name == 'HISTORY':
        return ViewStyle.HISTORY
    return ViewStyle.INVALID

def is_circle(name):
    """
    Returns True if given string parameter is for circle.
    :param name: name of figure.
    :return True or False.
    """
    return name.upper() == 'CIRCLE'

def is_ellipse(name):
    """
    Returns True if given string parameter is for ellipse.
    :param name: name of figure.
    :return True or False.
    """
    return name.upper() == 'ELLIPSE'

def is_paperclip(name):
    """
    Returns True if given string parameter is for paperclip curve.
    :param name: name of figure.
    :return True or False.
    """
    return name.upper() == 'PAPERCLIP'

def is_rectangle(name):
    """
    Returns True if given string parameter is for rectangle.
    :param name: name of figure.
    :return True or False.
    """
    return name.upper() == 'RECTANGLE'


def is_figure(param):
    """
    Detects if first command line argument is figure name.
    :param param: first command line argument
    :return: True if user specified figure name
    (like circle, ellipse, paperclip, rectangle).
    False otherwise.
    """
    return is_circle(param) or is_ellipse(param) or is_paperclip(param) or is_rectangle(param)


def cmp_curve(curve1, curve2):
    """
    Compares two curves to sort list of curves in descending order.
    :param curve1:
    :param curve2: 
    :return -1 if 1st curve is longer.
    """
    l1 = curve_ops.get_curve_size(curve1)
    l2 = curve_ops.get_curve_size(curve2)
    if l1 > l2:
        return -1
    if l1 < l2:
        return 1
    return 0


def get_curve_image_shape(args_width, args_height, curves):
    """
    Calculate width and height of image to display all curves.
    Also calculate horizontal and vertical displacements
    to display curves on image nicely.
    :param args_width: width for image specified by user
    :param args_height:  height for image specified by user
    :param curves: array of curves
    :return tuple of width and height of image and horizontal and vertical displacements.
    """
    if len(curves) == 0:
        return (0, 0, 0, 0)
    curve_cols = []
    curve_rows = []
    for curve in curves:
        curve_cols.append(int(min(curve[0])))
        curve_cols.append(int(max(curve[0])))
        curve_rows.append(int(min(curve[1])))
        curve_rows.append(int(max(curve[1])))

    horizontal_amplitude = max(curve_cols) - min(curve_cols)
    width = horizontal_amplitude if args_width == 0 else args_width
    cx = -min(curve_cols)
    if args_width == 0:
        cx += width // 8
        width += width // 4
    else:
        cx += (args_width - horizontal_amplitude) // 2

    vertical_amplitude = max(curve_rows) - min(curve_rows)
    height = vertical_amplitude if args_height == 0 else args_height
    cy = -min(curve_rows)
    if args_height == 0:
        cy += height // 8
        height += height // 4
    else:
        cy += (args_height - vertical_amplitude) // 2

    return (width, height, cx, cy)


def get_figure_curve(args):
    """
    Returns curve for figure with given parameters and number of points or empty curve.
    :param args: command line parameters.
    :return curve for specified figure (circle, ellipse, paperclip, rectangle).
    """
    if is_circle(args.image_path):
        return fig_curves.get_circle_curve(args.radius, args.num_points)

    if is_ellipse(args.image_path):
        return  fig_curves.get_ellipse_curve(args.radius_x, args.radius_y, args.num_points)

    if is_paperclip(args.image_path):
        return fig_curves.get_paperclip_curve(args.radius, args.num_points)

    if is_rectangle(args.image_path):
        return fig_curves.get_rectangle_curve(args.side_x, args.side_y, args.num_points)

    print("Invalid figure specified '", args.image_path, "'")
    return curve_ops.get_empty_curve()


#
# reads image from png file and runs curve shortening flow for it
# at every iteration curve is saved in folder as png file.
# Then those images are used to create movie.
#
def main():
    """
    Main function.
    """
    args = parse_command_line()
    if args is None:
        sys.exit(1)

    curves = []
    if is_figure(args.image_path):
        curve = get_figure_curve(args)
        if len(curve) == 0:
            sys.exit(1)
        curves.append(curve)
    else:
        # image_path is for image file
        if not os.path.exists(args.image_path):
            print("File '", args.image_path, "' doesn't exist!")
            sys.exit(1)

        image_path = args.image_path
        extent = get_extension(image_path)
        if not accept_extension(extent):
            print("File '", args.image_path, "' is not supported!")
            sys.exit(1)

        extension = '.' + extent

        # load original image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"

        # binarize image and set standard background and foreground colors
        img_ops.binarize(img)

        # perform median filtering
        median_img = cv2.medianBlur(img, args.median_filter) if args.median_filter > 0 else img
        if args.median_filter > 0:
            cv2.imwrite(image_path.replace(extension, "_median.png"), median_img)

        # perform curve thinning
        thinned_img = cv2.ximgproc.thinning(median_img)
        cv2.imwrite(image_path.replace(extension, "_thinned.png"), thinned_img)

        # extract curves from image
        num_extracted = 0
        for i in range(0, args.number_curves):
            curve = extract_curve(thinned_img)
            nc = curve_ops.get_curve_size(curve)
            if nc > 0:
                print("len=", nc)
                curves.append(curve)
                curve_img = img_ops.save_curve_to_image(curve)
                cv2.imwrite(image_path.replace(extension, "_extracted" + str(i + 1) + ".png"), curve_img)
                num_extracted += 1
            else:
                print("ERROR: Curve #", i + 1, " is not available")
                break

        print("-----------------------")

        if num_extracted != args.number_curves:
            print("ERROR: Number of extracted curves = ", num_extracted, " is less than ", args.number_curves)
            sys.exit(1)

        # sort curves by length in ascending order
        curves = sorted(curves, key=cmp_to_key(cmp_curve), reverse = args.ascending)
        for curve in curves:
            print("len=", curve_ops.get_curve_size(curve))

        if args.curve_no > 0:
            if args.curve_no > len(curves):
                print("Invalid number of curve=", args.curve_no, " - too big!")
                sys.exit(1)
            selected_curve = np.append(curve_ops.get_empty_curve(), curves[args.curve_no-1], axis=1)
            curves.clear()
            curves.append(selected_curve)

    width, height, cx, cy = get_curve_image_shape(args.width, args.height, curves)

    if not os.path.isdir(args.dest_folder):
        print("Folder '", args.dest_folder, "' doesn't exist!")
        sys.exit(1)

    if args.iterations <= 0:
        print("Invalid number of iterations!")
        sys.exit(1)

    view_style = get_view_style(args.view)

    if view_style == ViewStyle.SOLID:
        curve_colors = get_color_palette(args.color_palette)
        if len(curve_colors) == 0:
            print("Invalid color palette '", args.color_palette, "'")
            sys.exit(1)

        background_color = get_background_color(args.color_palette)
        if len(background_color) == 0:
            print("Invalid color palette '", args.color_palette, "'")
            sys.exit(1)
        history_colors = []
    else:
        foreground_color = color_ops.convert_to_rgb(args.fg_color)
        curve_colors = [foreground_color, foreground_color, foreground_color, foreground_color, foreground_color]
        background_color = color_ops.convert_to_rgb(args.bg_color)

        if view_style == ViewStyle.HISTORY:
            history_colors = get_history_colors(foreground_color, background_color, HistoryViewStyle.MAXCOUNT)
        else:
            history_colors = []

    i = 0
    for curve in curves:
        curve = geom.translate(curve, cx, cy)
        # perform smoothing of extracted curve to compensate curve drawing singularities
        if not is_figure(args.image_path):
            curve = geom.smoothen_curve(curve, 3, 1, 100)
        flow = cs.CurveShortener()
        if args.preserve_area:
            flow.set_preserve_area()
        if args.save_info:
            flow.set_save_additional_info()
        callback_fcn = cb.vector_view_callback if view_style == ViewStyle.VECTOR else (
            cb.solid_view_callback if view_style == ViewStyle.SOLID else (cb.contour_view_callback if view_style == ViewStyle.CONTOUR
            else cb.history_view_callback))
        flow.setCallBack(callback_fcn,
                         (height, width, args.dest_folder, args.iterations, args.save_every_n, background_color,
                          curve_colors[i], history_colors))
        flow.run(curve)
        i += 1


if __name__ == "__main__":
    main()
