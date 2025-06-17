import sys
import os
import argparse
import math
import cv2
import numpy as np
import geometry as geom
import CurveExtractor as ce
import CurveShortener as cs
import image_operations as img_ops
import color_operations as color_ops

def parse_command_line():
    parser = argparse.ArgumentParser(prog='shorten_curve.py', description='Curve shortening flow demo', epilog='Text at the bottom of help')
    parser.add_argument('imagePath', help='source image file containing closed simple curve')
    parser.add_argument('destFolder', help='Folder path to store output images')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=100, help='max number of iterations')
    parser.add_argument('-p', '--preserve_length', required=False, action='store_true', help='preserve length')
    parser.add_argument('-s', '--save_every_n', type=int, required=False, default=5, help='How often image with curve is saved. Default is 5 - every 5th image is saved')
    parser.add_argument('-v', '--vector_look', required=False, action='store_true', help='Display curve shortening flow vectors on curve')
    parser.add_argument('-m', '--median_filter', type=int, required=False, default=0, choices=[3,5,7,9], help='apply median filter for source image with windowsize n')
    parser.add_argument('-n', '--number_curves', type=int, required=False, default=1, choices=[1,2,3,4,5], help='sets number of curves to apply shortening flow')
    parser.add_argument('-c', '--color_palette', type=str, required=False, default='blue', choices=['blue','green','red'], help='sets coloring for visualization of filled curves')
    return parser.parse_args()

def get_extension(filePath):
    ext = filePath.rpartition('.')[-1]
    return ext

def accept_extension(extent):
    ext = extent.strip().lower()
    if len(ext) == 0: return False
    return len([s for s in ["bmp", "png", "jpg", "jpeg"] if s == ext]) == 1


# callback function returns True to terminate flow, False otherwise
def myCallBackVectorLook(curve, curvature, iter, is_circle, obj):
    print("iter=", iter, " curve arc length=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[4]
        if iter % n == 0:
            rows = obj[0]
            cols = obj[1]
            path = obj[2]
            file_path = os.path.join(path, 'image' + (str(iter)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img =  cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) if image_exists else np.zeros((rows, cols), np.uint8)
            if not img is None:
                img_ops.draw_curve_lines(img, curve)
                normal_field = geom.get_normal_field(curve)
                normal_unit_field = geom.normalize(normal_field)
                img_ops.display_shortening_field(img, curve, curvature, normal_unit_field)
                cv2.imwrite(file_path, img)
            
        max_iterations = obj[3]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations>0 and iter==max_iterations) or (max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)
    return True

def myCallBackClassicLook(curve, curvature, iter, is_circle, obj):
    print("iter=", iter, " curve arclength=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[4]
        if iter % n == 0:
            rows = obj[0]
            cols = obj[1]
            path = obj[2]
            file_path = os.path.join(path, 'image' + (str(iter)).zfill(5) + '.png')
            image_exists = os.path.exists(file_path)
            img =  cv2.imread(file_path, cv2.IMREAD_COLOR) if image_exists else np.zeros((rows, cols, 3), np.uint8)
            
            cv2.floodFill(img, None, (1, 1), obj[5])
            if not img is None:
                img_ops.draw_curve_lines(img, curve, obj[6])
                if min(curvature) >= 0:
                    img_ops.fill_convex_curve(img, curve, obj[6])
                else:
                    img_ops.fill_curve(img, curve, curvature, obj[6])
                cv2.imwrite(file_path, img)

        max_iterations = obj[3]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations>0 and iter==max_iterations) or (max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)
        
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
    curve = geom.move_curve_center(curve, (cols/2, rows/2))
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
    min_cols = []
    max_cols = []
    min_rows = []
    max_rows = []
    num_extracted = 0
    for i in range(0, args.number_curves):
        curve = extract_curve(thinned_img)
        if len(curve) > 0:
            print("len=", len(curve))
            curves.append(curve)
            min_cols.append(min([p[0] for p in curve]))
            min_rows.append(min([p[1] for p in curve]))
            max_cols.append(max([p[0] for p in curve]))
            max_rows.append(max([p[1] for p in curve]))
            curve_img = save_curve_to_image(curve)
            cv2.imwrite(imagePath.replace(extension, "_extracted" + str(i+1) + ".png"), curve_img)
            num_extracted += 1
        else:
            print("ERROR: Curve #", i+1, " is not available")
            break

    if num_extracted != args.number_curves:
        print("ERROR: Number of extracted curves = ", num_extracted, " is less than ". args.number_curves)
        return
    
    width = max(max_cols) - min(min_cols)
    dx = width // 10 - min(min_cols)
    width += width // 5
    height = max(max_rows) - min(min_rows)
    dy = height // 10 - min(min_rows)
    height += height // 5
    
    if not args.vector_look:
        curve_colors = get_color_palette(args.color_palette)
        if len(curve_colors) == 0:
            print("Invalid color palette '", args.color_palette, "'")
            exit(1)

        background_color = get_background_color(args.color_palette)
        if len(background_color) == 0:
            print("Invalid color palette '", args.color_palette, "'")
            exit(1)
    else:
        curve_colors = []
        background_color = ()


    for i in range(0, len(curves)):
        # perform smoothing of extracted curve to compensate drawing singularities
        curve = geom.translate(curves[i], (dx, dy))
        curve = geom.smoothen_curve(curve, 3, 1, 100)
        flow = cs.CurveShortener()
        if args.preserve_length: 
            flow.set_preserve_curve_length()
        flow.set_save_additional_info()
        flow.setCallBack(myCallBackVectorLook if args.vector_look else myCallBackClassicLook, (height, width, args.destFolder, args.iterations, args.save_every_n, background_color, curve_colors[i]))
        flow.run(curve)
   

if __name__ == "__main__":
    main()