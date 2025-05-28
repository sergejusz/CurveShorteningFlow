import math
import cv2
import numpy as np
import geometry as geom
import list_operations as list_ops

def get_signal_color():
    return 255

def get_background_color():
    return 0

def fill_image(img,  color):
    rows,cols = img.shape
    return cv2.rectangle(img, (0,0), (cols-1, rows-1), color, -1)

def draw_curve_points_zoom(img, curve, color=get_signal_color(), magnify=1):
    if len(curve) == 0: return

    rows,cols = img.shape
    
    x_center, y_center = geom.get_curve_center(curve)

    for p in curve:
        x = int((p[0]-x_center)*magnify+int(cols/2))
        y = int((p[1]-y_center)*magnify+int(rows/2))
        if x >=0 and x < cols and y >=0 and y < rows:
            img[y,x] = color

def draw_curve_points(img, curve, color=get_signal_color()):
    rows,cols = img.shape
    for p in curve:
        x = int(p[0])
        y = int(p[1])
        if x >=0 and x < cols and y >=0 and y < rows:
            img[y,x] = color
                    
def draw_curve_lines(img, curve, color=get_signal_color()):
    rows,cols = img.shape
    n = len(curve)
    p1 = curve[0]
    for i in range(1,n+1):
        p2 = curve[i % n]
        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color, 1)
        p1 = p2

def display_vector_field(img, curve, vectorField, color=get_signal_color(), delta_n = 5, magnify = 20):
    N = len(curve)
    i = 0
    while i < N:
        p = curve[i]
        v = vectorField[i]
        cv2.line(img, (int(p[0]),int(p[1])), (int(p[0]-magnify*v[0]),int(p[1]-magnify*v[1])), color, 1)
        i += delta_n
        
def display_shortening_field(img, curve, curvature, vectorField, color=get_signal_color(), delta_n = 5, magnify = 20):
    N = len(curve)
    maxc = max([math.fabs(c) for c in curvature])
    d = 1.0/maxc
    i = 0
    while i < N:
        p = curve[i]
        v = vectorField[i]
        c = curvature[i]*d
        cv2.line(img, (int(p[0]),int(p[1])), (int(p[0]-magnify*c*v[0]),int(p[1]-magnify*c*v[1])), color, 1)
        i += delta_n

def create_curve_image(curve):
    curve_width = geom.get_horizontal_amplitude(curve)
    curve_height = geom.get_vertical_amplitude(curve)
    image_width = curve_width + curve_width // 2
    image_height = curve_height + curve_height // 2
    new_image = np.zeros((image_height, image_width), np.uint8)
    return new_image
    
# detect background and foreground colors in image and 
# turn it to image with black bakgound and white signal pixels.
def binarize(image):
    if image is None:
         return

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    backgr_color = np.argmax(hist)
    hist[backgr_color] = 0
    foregr_color = np.argmax(hist)

    rows,cols = image.shape
        
    for row in range(rows):
            for col in range(cols):
                if image[row, col] != backgr_color:
                    image[row, col] = foregr_color
                    
    # change colors for standard color scheme: 
    # 0 --> background, 255 --> signal color.
    if backgr_color == get_background_color() and foregr_color == get_signal_color():
        return
            
    for row in range(rows):
        for col in range(cols):
            if image[row, col] == backgr_color:
                image[row, col] = get_background_color()
            else:
                if image[row, col] == foregr_color:
                    image[row, col] = get_signal_color()

