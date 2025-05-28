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

def parse_command_line():
    parser = argparse.ArgumentParser(prog='shorten_curve.py', description='Curve shortening flow demo', epilog='Text at the bottom of help')
    parser.add_argument('imagePath', help='source image file containing closed simple curve')
    parser.add_argument('destFolder', help='Folder path to store output images')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=100, help='max number of iterations')
    parser.add_argument('-p', '--preserve_length', required=False, action='store_true', help='preserve length')
    parser.add_argument('-s', '--save_every_n', type=int, required=False, default=5, help='How often image with curve is saved. Default is 5 - every 5th image is saved')
    parser.add_argument('-v', '--vectors', required=False, action='store_true', help='Display curve shortening flow vectors on curve')
    parser.add_argument('-m', '--median_filter', type=int, required=False, default=0, help='apply median filter for source image with windowsize n')
    return parser.parse_args()

def get_extension(filePath):
    ext = filePath.rpartition('.')[-1]
    return ext

def accept_extension(extent):
    ext = extent.strip().lower()
    if len(ext) == 0: return False
    return len([s for s in ["bmp", "png", "jpg", "jpeg"] if s == ext]) == 1


# callback function returns True to terminate flow, False otherwise
def myCallBackVectorLook(curve, curvature, iter, obj):
    print("iter=", iter, " curve arc length=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[4]
        if iter % n == 0:
            rows = obj[0]
            cols = obj[1]
            img = np.zeros((rows, cols), np.uint8)
            
            img_ops.draw_curve_lines(img, curve)
        
            normal_field = geom.get_normal_field(curve)
            normal_unit_field = geom.normalize(normal_field)
        
            img_ops.display_shortening_field(img, curve, curvature, normal_unit_field)
        
            path = obj[2]
            cv2.imwrite(os.path.join(path, 'image' + (str(iter)).zfill(5) + '.png'), img)
            
        max_iterations = obj[3]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations>0 and iter==max_iterations) or (max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)
    return True

def myCallBackClassicLook(curve, curvature, iter, obj):
    print("iter=", iter, " curve arclength=", geom.get_curve_length(curve))

    if obj is not None:
        n = obj[4]
        if iter % n == 0:
            rows = obj[0]
            cols = obj[1]

            img = np.zeros((rows, cols), np.uint8)
            img = img_ops.fill_image(img, 150)
            img_ops.draw_curve_lines(img, curve, 255)
            cv2.floodFill(img, None, (0, 0), 255)

            path = obj[2]
            cv2.imwrite(os.path.join(path, 'image' + (str(iter)).zfill(5) + '.png'), img)

        max_iterations = obj[3]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (max_iterations>0 and iter==max_iterations) or (max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)
        
    return True

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
    
    # extract curve from image
    signalColor = img_ops.get_signal_color()
    curveExtractor = ce.CurveExtractor()
    curve = curveExtractor.extract(thinned_img, signalColor)
    if len(curve) == 0:
        print("curveExtractor.extract returned empty curve")
        return

    curve_img = img_ops.create_curve_image(curve)
    rows,cols = curve_img.shape
    curve = geom.move_curve_center(curve, (cols/2, rows/2))
    img_ops.draw_curve_points(curve_img, curve)
    cv2.imwrite(imagePath.replace(extension, "_extracted.png"), curve_img)

    # perform smoothing of extracted curve to compensate drawing singularities
    curve = geom.smoothen_curve(curve, 3, 1, 100)

    flow = cs.CurveShortener()
    if args.preserve_length: 
        flow.set_preserve_curve_length()
    flow.set_save_additional_info()
    flow.setCallBack(myCallBackVectorLook if args.vectors else myCallBackClassicLook, (rows, cols, args.destFolder, args.iterations, args.save_every_n))
    flow.run(curve)
   

if __name__ == "__main__":
    main()