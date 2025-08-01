import sys
import os
from enum import IntEnum
import argparse
import math
import cv2
import numpy as np
import geometry as geom
import CurveExtractor as ce
import CurveShortener as cs
import image_operations as img_ops
import color_operations as color_ops
import curve_operations as curve_ops


class ViewStyle(IntEnum):
    Invalid = -1
    Contour = 1
    Solid = 2
    Vector = 3


class CallbackArgs(IntEnum):
    Rows = 0
    Cols = 1
    Path = 2
    MaxIterations = 3
    SaveToFileCounter = 4
    BackgroundColor = 5
    ForegroundColor = 6


def parse_command_line():
    parser = argparse.ArgumentParser(prog='shorten_curve.py', description='Curve shortening flow demo',
                                     epilog='Text at the bottom of help')
    parser.add_argument('imagePath', help='source image file containing closed simple curve')
    parser.add_argument('destFolder', help='Folder path to store output images')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=100, help='max number of iterations')
    parser.add_argument('-p', '--preserve_area', required=False, action='store_true', help='preserve area')
    parser.add_argument('-s', '--save_every_n', type=int, required=False, default=5,
                        help='How often image with curve is saved. Default is 5 - every 5th image is saved')
    parser.add_argument('-v', '--view', required=False, default='contour', choices=['contour', 'solid', 'vector'],
                        help='View style of curves under flow')
    parser.add_argument('-m', '--median_filter', type=int, required=False, default=0, choices=[3, 5, 7, 9],
                        help='apply median filter for source image with windowsize n')
    parser.add_argument('-n', '--number_curves', type=int, required=False, default=1, choices=[1, 2, 3, 4, 5],
                        help='sets number of curves to apply shortening flow')
    parser.add_argument('-c', '--color_palette', type=str, required=False, default='blue',
                        choices=['blue', 'green', 'red'], help='sets coloring for visualization of filled curves')
    parser.add_argument('-a', '--save_add_info', required=False, action='store_true',
                        help='save additional information')
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
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) if image_exists else np.zeros((rows, cols), np.uint8)
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
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.zeros((rows, cols, 3), np.uint8)
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj[CallbackArgs.ForegroundColor])
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
    return ViewStyle.Invalid


#
# reads image from png file and runs curve shortening flow for it
# at every iteration curve is saved in folder as png file.
# Then those images are used to create movie.
#
def main():
    args = parse_command_line()
    if args is None: exit

    if not os.path.exists(args.imagePath):
        print("File '", args.imagePath, "' doesn't exist!")
        exit(1)

    imagePath = args.imagePath
    extent = get_extension(imagePath)
    if not accept_extension(extent):
        print("File '", args.imagePath, "' is not supported!")
        exit(1)

    extension = '.' + extent

    if not os.path.isdir(args.destFolder):
        print("Folder '", args.destFolder, "' doesn't exist!")
        exit(1)

    if args.iterations <= 0:
        print("Invalid number of iterations!")
        exit(1)

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
    curves = []
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

    if num_extracted != args.number_curves:
        print("ERROR: Number of extracted curves = ", num_extracted, " is less than ", args.number_curves)
        return

    width = max(curve_cols) - min(curve_cols)
    dx = width // 10 - min(curve_cols)
    width += width // 5
    height = max(curve_rows) - min(curve_rows)
    dy = height // 10 - min(curve_rows)
    height += height // 5

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
    else:
        white_color = (255, 255, 255)
        curve_colors = [white_color, white_color, white_color, white_color, white_color]
        background_color = ()

    for i in range(0, len(curves)):
        # perform smoothing of extracted curve to compensate drawing singularities
        curve = geom.translate(curves[i], dx, dy)
        curve = geom.smoothen_curve(curve, 3, 1, 100)
        flow = cs.CurveShortener()
        if args.preserve_area:
            flow.set_preserve_area()
        if args.save_add_info:
            flow.set_save_additional_info()
        callBackFcn = myCallBackVectorView if view_style == ViewStyle.Vector else (
            myCallBackSolidView if view_style == ViewStyle.Solid else myCallBackContourView)
        flow.setCallBack(callBackFcn,
                         (height, width, args.destFolder, args.iterations, args.save_every_n, background_color,
                          curve_colors[i]))
        flow.run(curve)


if __name__ == "__main__":
    main()
