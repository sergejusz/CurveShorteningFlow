import sys
import os
from enum import IntEnum
import argparse
import math
from functools import cmp_to_key
import cv2
import numpy as np
import geometry as geom
import CurveExtractor as ce
import CurveShortener as cs
import image_operations as img_ops
import color_operations as color_ops
import curve_operations as curve_ops
import ColorInterpolator as ci

class ViewStyle(IntEnum):
    Invalid = -1
    Contour = 1
    Solid = 2
    Vector = 3
    History = 4


class CallbackArgs(IntEnum):
    Rows = 0
    Cols = 1
    Path = 2
    MaxIterations = 3
    SaveToFileCounter = 4
    BackgroundColor = 5
    ForegroundColor = 6
    HistoryColors = 7

class HistoryViewStyle(IntEnum):
    MaxCount = 20
    SkipIterations = 100


def parse_command_line():
    parser = argparse.ArgumentParser(prog='shorten_curve.py', description='Curve shortening flow demo',
                                     epilog='Text at the bottom of help')
    parser.add_argument('imagePath', help='source image file containing closed simple curve or CIRCLE or ELLIPSE or PAPERCLIP curve')
    parser.add_argument('destFolder', help='Folder path to store output images')
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
    parser.add_argument('--save_info', required=False, action='store_true',
                        help='save additional information to files')
    parser.add_argument('--radius', type=int, required=False, default=0, help='circle radius')
    parser.add_argument('--radius_x', type=int, required=False, default=0, help='ellipse radius for x axis')
    parser.add_argument('--radius_y', type=int, required=False, default=0, help='ellipse radius for y axis')
    parser.add_argument('--ascending', required=False, action='store_true', help='process multiple curves from shortest to longest. Default order is from longest to shortest.')
    parser.add_argument('--curve_no', type=int, required=False, default=0, choices={1,2,3,4,5}, help='process only specified curve with given number. Number starts from 1.')
    parser.add_argument('--bg_color', type=str, required=False, default='000000', help='background color for contour, vector and history views')
    parser.add_argument('--fg_color', type=str, required=False, default='ffffff', help='foreground color for contour, vector and history views')
    return parser.parse_args()


def get_extension(filePath):
    ext = filePath.rpartition('.')[-1]
    return ext


def accept_extension(extent):
    ext = extent.strip().lower()
    if len(ext) == 0: return False
    return len([s for s in ["bmp", "png", "jpg", "jpeg"] if s == ext]) == 1


def get_sample_skip_count_for_vector(curve_length, num_samples):
    ds = curve_length / num_samples
    if ds < 0.1: return 30
    if ds < 0.2: return 20
    if ds < 0.5: return 10
    if ds < 1.0: return 5
    if ds < 2.0: return 2

    return 1

def get_vector_scaling_factor(curve):
    ampl = min(geom.get_curve_amplitude(curve))
    if ampl >= 60: return 20.0
    if ampl >= 40: return 12.0
    if ampl >= 20: return 0.2*ampl + 2.0
    if ampl >= 10: return 0.3*ampl
    return 3.0


def get_history_colors(fg_color, bg_color, num_levels):
    color_interpolator = ci.ColorInterpolator(fg_color, bg_color, num_levels)
    history_colors = []
    for i in range(num_levels):
        (r, g, b) = color_interpolator.calculate(i)
        history_colors.append((int(r), int(g), int(b)))
    return history_colors

# callback function returns True to terminate flow, False otherwise
def myCallBackVectorView(curve, curvature, curve_length, iter, is_circle, obj):
    print("iter=", iter, " curve arc length=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[CallbackArgs.SaveToFileCounter]
        if iter % n == 0:
            rows = obj[CallbackArgs.Rows]
            cols = obj[CallbackArgs.Cols]
            file_path = os.path.join(obj[CallbackArgs.Path], 'image' + (str(iter)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.full((rows, cols, 3), obj[CallbackArgs.BackgroundColor], np.uint8)
            if not img is None:
                img_ops.draw_curve_lines(img, curve)
                normal_field = geom.get_normal_field(curve)
                normal_unit_field = geom.normalize(normal_field)
                skip_samples = get_sample_skip_count_for_vector(curve_length, curve_ops.get_curve_size(curve))
                scaling_factor = get_vector_scaling_factor(curve)
                img_ops.display_shortening_field(img, curve, curvature, normal_unit_field,
                                                 obj[CallbackArgs.ForegroundColor], delta_n=skip_samples, magnify = scaling_factor)
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MaxIterations]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations > 0 and iter == max_iterations) or (
                    max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)
    return True


def myCallBackContourView(curve, curvature, curve_length, iter, is_circle, obj):
    print("iter=", iter, " curve arclength=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[CallbackArgs.SaveToFileCounter]
        if iter % n == 0:
            rows = obj[CallbackArgs.Rows]
            cols = obj[CallbackArgs.Cols]
            file_path = os.path.join(obj[CallbackArgs.Path], 'image' + (str(iter)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.full((rows, cols, 3), obj[CallbackArgs.BackgroundColor], np.uint8)
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj[CallbackArgs.ForegroundColor])
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MaxIterations]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations > 0 and iter == max_iterations) or (
                    max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)

    return True

def myCallBackHistoryView(curve, curvature, curve_length, iter, is_circle, obj, curves = []):
    print("iter=", iter, " curve arclength=", geom.get_curve_length(curve))


    if iter == 0: curves.clear()

    if obj is not None:
        if iter % HistoryViewStyle.SkipIterations == 0:
            curves.append(curve)
            if len(curves) > HistoryViewStyle.MaxCount:
                del curves[0]

        n = obj[CallbackArgs.SaveToFileCounter]
        if iter % n == 0:
            rows = obj[CallbackArgs.Rows]
            cols = obj[CallbackArgs.Cols]
            file_path = os.path.join(obj[CallbackArgs.Path], 'image' + (str(iter)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.full((rows, cols, 3), obj[CallbackArgs.BackgroundColor], np.uint8)

            if not img is None:
                m = len(curves)
                i = 0
                for curve in curves:
                    img_ops.draw_curve_lines(img, curve, obj[CallbackArgs.HistoryColors][m - i - 1])
                    i += 1
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MaxIterations]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations > 0 and iter == max_iterations) or (
                    max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)

    return True


def myCallBackSolidView(curve, curvature, curve_length, iter, is_circle, obj):
    print("iter=", iter, " curve arclength=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[CallbackArgs.SaveToFileCounter]
        if iter % n == 0:
            rows = obj[CallbackArgs.Rows]
            cols = obj[CallbackArgs.Cols]
            file_path = os.path.join(obj[CallbackArgs.Path], 'image' + (str(iter)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.zeros((rows, cols, 3), np.uint8)

            cv2.floodFill(img, None, (1, 1), obj[CallbackArgs.BackgroundColor])
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj[CallbackArgs.ForegroundColor])
                if min(curvature) >= 0:
                    img_ops.fill_convex_curve(img, curve, obj[CallbackArgs.ForegroundColor])
                else:
                    img_ops.fill_curve(img, curve, curvature, obj[CallbackArgs.ForegroundColor])
                cv2.imwrite(file_path, img)

        max_iterations = obj[CallbackArgs.MaxIterations]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations > 0 and iter == max_iterations) or (
                    max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)

    return True


def extract_curve(img):
    curveExtractor = ce.CurveExtractor()
    signal_color = img_ops.get_signal_color()
    curve = curveExtractor.extract(img, signal_color)
    # remove extracted curve from image
    img_ops.clear_curve(img, curve, 5, 5, 0)
    return curve


def save_curve_to_image(curve):
    img = img_ops.create_curve_image(curve)
    rows, cols = img.shape
    curve = geom.move_curve_center(curve, cols / 2, rows / 2)
    img_ops.draw_curve_points(img, curve)
    return img


def get_color_palette(coloring_name):
    upper_name = coloring_name.upper()
    if upper_name == 'RED':
        return color_ops.get_red_palette()
    if upper_name == 'BLUE':
        return color_ops.get_blue_palette()
    if upper_name == 'GREEN':
        return color_ops.get_green_palette()
    return []


def get_background_color(coloring_name):
    upper_name = coloring_name.upper()
    if upper_name == 'RED':
        return color_ops.get_red_background_color()
    if upper_name == 'BLUE':
        return color_ops.get_blue_background_color()
    if upper_name == 'GREEN':
        return color_ops.get_green_background_color()
    return ()


def get_view_style(view_style):
    upper_name = view_style.upper()
    if upper_name == 'CONTOUR':
        return ViewStyle.Contour
    if upper_name == 'SOLID':
        return ViewStyle.Solid
    if upper_name == 'VECTOR':
        return ViewStyle.Vector
    if upper_name == 'HISTORY':
        return ViewStyle.History
    return ViewStyle.Invalid

def is_circle(param):
    return param.upper() == 'CIRCLE'

def is_ellipse(param):
    return param.upper() == 'ELLIPSE'

def is_paperclip(param):
    return param.upper() == 'PAPERCLIP'

def is_figure(param):
    return is_circle(param) or is_ellipse(param) or is_paperclip(param)

def cmp_curve(curve1, curve2):
    s1 = geom.get_curve_length(curve1)
    s2 = geom.get_curve_length(curve2)
    if s1 > s2:
        return -1
    elif s1 < s2:
        return 1
    return 0

#
# reads image from png file and runs curve shortening flow for it
# at every iteration curve is saved in folder as png file.
# Then those images are used to create movie.
#
def main():
    args = parse_command_line()
    if args is None: exit

    width = 0
    height = 0
    dx = 0
    dy = 0
    curves = []

    if is_figure(args.imagePath):
        if is_circle(args.imagePath):
           if args.radius == 0:
                print("ERROR: radius cannot be zero")
                exit(1)

           num_points = args.radius * 4
           num_points += args.radius // 10

           width = args.radius * 2
           width += width // 5
           height = width
           curves.append(curve_ops.get_circle(width / 2, height / 2, args.radius, num_points))
        elif is_ellipse(args.imagePath):
            if args.radius_x == 0 or args.radius_y == 0:
                print("ERROR: radius cannot be zero")
                exit(1)

            num_points = args.radius_x * 2 + args.radius_y * 2
            num_points += (args.radius_x + args.radius_y) // 5
            width = args.radius_x * 2
            width += width // 5
            height = args.radius_y * 2
            height += height // 5
            curves.append(curve_ops.get_ellipse(width / 2, height / 2, args.radius_x, args.radius_y, num_points))

        elif is_paperclip(args.imagePath):
           if args.radius == 0:
                print("ERROR: radius cannot be zero")
                exit(1)

           num_points = args.radius * 8

           width = args.radius * 5
           width += width // 2
           height = args.radius * 2
           # height made bigger because for 'calculation' errors flat sides of paperclip
           # expand to the top/bottom borders of image
           height += height // 4
           curves.append(curve_ops.get_paperclip(args.radius + args.radius / 2, height / 2, args.radius, num_points))

        else:
            print("Invalid figure specified '", args.imagePath, "'")
            exit(1)
    else:
        # imagePath is for image file
        if not os.path.exists(args.imagePath):
            print("File '", args.imagePath, "' doesn't exist!")
            exit(1)

        imagePath = args.imagePath
        extent = get_extension(imagePath)
        if not accept_extension(extent):
            print("File '", args.imagePath, "' is not supported!")
            exit(1)

        extension = '.' + extent

        # load original image
        img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"

        # binarize image and set standard background and foreground colors
        img_ops.binarize(img)

        # perform median filtering
        median_img = cv2.medianBlur(img, args.median_filter) if args.median_filter > 0 else img
        if args.median_filter > 0:
            cv2.imwrite(imagePath.replace(extension, "_median.png"), median_img)

        # perform curve thinning
        thinned_img = cv2.ximgproc.thinning(median_img)
        cv2.imwrite(imagePath.replace(extension, "_thinned.png"), thinned_img)

        # extract curves from image
        curve_cols = []
        curve_rows = []
        num_extracted = 0
        for i in range(0, args.number_curves):
            curve = extract_curve(thinned_img)
            nc = curve_ops.get_curve_size(curve)
            if nc > 0:
                print("len=", nc)
                curves.append(curve)
                curve_cols.append(int(min(curve[0])))
                curve_cols.append(int(max(curve[0])))
                curve_rows.append(int(min(curve[1])))
                curve_rows.append(int(max(curve[1])))
                curve_img = save_curve_to_image(curve)
                cv2.imwrite(imagePath.replace(extension, "_extracted" + str(i + 1) + ".png"), curve_img)
                num_extracted += 1
            else:
                print("ERROR: Curve #", i + 1, " is not available")
                break

        print("-----------------------")

        if num_extracted != args.number_curves:
            print("ERROR: Number of extracted curves = ", num_extracted, " is less than ", args.number_curves)
            return

        # sort curves by length in ascending order
        curves = sorted(curves, key=cmp_to_key(cmp_curve), reverse = args.ascending)
        for curve in curves:
            print("len=", curve_ops.get_curve_size(curve))

        if args.curve_no != 0:
            if args.curve_no > len(curves):
                print("Invalid number of curve=", args.curve_no, " - too big!")
                exit(1)
            selected_curve = curves[args.curve_no]
            curve_cols.clear()
            curve_rows.clear()
            curve_cols.append(int(min(selected_curve[0])))
            curve_cols.append(int(max(selected_curve[0])))
            curve_rows.append(int(min(selected_curve[1])))
            curve_rows.append(int(max(selected_curve[1])))

        width = max(curve_cols) - min(curve_cols)
        dx = width // 10 - min(curve_cols)
        width += width // 5
        height = max(curve_rows) - min(curve_rows)
        dy = height // 10 - min(curve_rows)
        height += height // 5



    if not os.path.isdir(args.destFolder):
        print("Folder '", args.destFolder, "' doesn't exist!")
        exit(1)

    if args.iterations <= 0:
        print("Invalid number of iterations!")
        exit(1)

    view_style = get_view_style(args.view)

    if view_style == ViewStyle.Solid:
        curve_colors = get_color_palette(args.color_palette)
        if len(curve_colors) == 0:
            print("Invalid color palette '", args.color_palette, "'")
            exit(1)

        background_color = get_background_color(args.color_palette)
        if len(background_color) == 0:
            print("Invalid color palette '", args.color_palette, "'")
            exit(1)
        history_colors = []
    else:
        foreground_color = color_ops.convert_to_rgb(args.fg_color)
        curve_colors = [foreground_color, foreground_color, foreground_color, foreground_color, foreground_color]
        background_color = color_ops.convert_to_rgb(args.bg_color)

        if view_style == ViewStyle.History:
            history_colors = get_history_colors(foreground_color, background_color, HistoryViewStyle.MaxCount)
        else:
            history_colors = []

    for i in range(0, len(curves)):
        curve = geom.translate(curves[i], dx, dy)
        # perform smoothing of extracted curve to compensate curve drawing singularities
        if not is_figure(args.imagePath):
            curve = geom.smoothen_curve(curve, 3, 1, 100)
        flow = cs.CurveShortener()
        if args.preserve_area:
            flow.set_preserve_area()
        if args.save_info:
            flow.set_save_additional_info()
        callBackFcn = myCallBackVectorView if view_style == ViewStyle.Vector else (
            myCallBackSolidView if view_style == ViewStyle.Solid else (myCallBackContourView if view_style == ViewStyle.Contour
            else myCallBackHistoryView))
        flow.setCallBack(callBackFcn,
                         (height, width, args.destFolder, args.iterations, args.save_every_n, background_color,
                          curve_colors[i], history_colors))
        flow.run(curve)


if __name__ == "__main__":
    main()
