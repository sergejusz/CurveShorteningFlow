import sys
import os
import math
import cv2
import numpy as np
import geometry as geom
import CurveExtractor as ce
import CurveShortener as cs
import image_operations as img_ops


# callback function returns True to terminate flow, False otherwise
def myCallBack(curve, curvature, iter, obj):
    print("iter=", iter, " curve arc length=", geom.get_curve_length(curve))

    if obj is not None:
        if iter % 5 == 0:
            img = np.zeros((obj[0], obj[1]), np.uint8)
            
            img_ops.draw_curve_lines(img, curve)
        
            normal_field = geom.get_normal_field(curve)
            normal_unit_field = geom.normalize(normal_field)
        
            img_ops.display_shortening_field(img, curve, curvature, normal_unit_field)
        
            path = obj[2]
            cv2.imwrite(os.path.join(path, 'image' + (str(iter)).zfill(5) + '.png'), img)
            
        iterations = obj[3]
        # terminate flow if number of iterations is exhausted or curve size in horizontal and vertical directions is small
        return (iterations>0 and iter==iterations) or (max(geom.get_horizontal_amplitude(curve), geom.get_vertical_amplitude(curve)) < 10)
    return True
#
# reads image from png file and runs curve shortening flow for it
# at every iteration curve is saved in folder as png file.
# Then those images are used to create movie.
#
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("python shorten_curve.py inputImagePath outputFolderPath [number_of_iterations]")
        return

    if not os.path.exists(args[0]):
        print("File '", args[0], "' doesn't exist!")
        return

    imagePath = args[0]

    # load original image
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    if not os.path.isdir(args[1]):
        print("Folder '", args[1], "' doesn't exist!")
        return

    image_folder = args[1]
    
    # number of iterations
    iterations = -1
    if len(args) >= 2: iterations = int(args[2])
    
    # perform median filtering with window 3x3
    median_img = cv2.medianBlur(img, 3)
    cv2.imwrite(imagePath.replace(".png", "_median.png"), median_img)
    
    # perform curve thinning
    thinned_img = cv2.ximgproc.thinning(median_img)
    cv2.imwrite(imagePath.replace(".png", "_thinned.png"), thinned_img)
    
    # extract curve from image
    signalColor = img_ops.get_signal_color()
    curveExtractor = ce.CurveExtractor()
    curve = curveExtractor.extract(thinned_img, signalColor)
    if len(curve) == 0:
        print("curveExtractor.extract returned empty curve")
        return

    rows, cols = img.shape
    sz = max(rows, cols)
    curve = geom.move_curve_center(curve, (cols/2, rows/2))
    curve_img = np.zeros((sz, sz), np.uint8)
    img_ops.draw_curve_points(curve_img, curve)
    cv2.imwrite(imagePath.replace(".png", "_extracted.png"), curve_img)

    # perform smoothing of extracted curve to compensate drawing singularities
    curve = geom.smoothen_curve(curve, 3, 1, 100)

    flow = cs.CurveShortener()
    # preserve curve length
    flow.set_preserve_curve_length()
    flow.set_save_additional_info()
    flow.setCallBack(myCallBack, (rows, cols, image_folder, iterations, False))
    flow.run(curve)
   

if __name__ == "__main__":
    main()