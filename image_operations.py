import math
import cv2
import numpy as np
from scipy import signal
import geometry as geom
import list_operations as list_ops
import collections

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
    attrs = img.shape
    rows = attrs[0]
    cols = attrs[1]
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


def draw_cross(img, x, y, color):
    rows, cols = img.shape[:2]
    cv2.line(img, (x-5, y), (x+5, y), color, 1)
    cv2.line(img, (x, y-5), (x, y+5), color, 1)
    
def fill_convex_curve(img, curve, color):
    if len(curve) == 0: return
    pc = geom.get_curve_center(curve)
    cv2.floodFill(img, None, (int(pc[0]), int(pc[1])), color)


# returns True if given color presents in 3x3 neighbourhood of (x,y) for given image
def has_color_in_neighborhood(img, x, y, color):
    is_scalar = isinstance(color, int)
    dpos =[(-1,-1), (-1,0), (-1,1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1,1)]
    for dp in dpos:
        img_color = img[y+dp[0], x+dp[1]]
        if is_scalar:
            has_color = img_color == color
        else:
            has_color = img_color[0] == color[0] and img_color[1] == color[1] and img_color[2] == color[2]
        if has_color:
            return True
    return False


def fill_curve(img, curve, curvature, color):
    if max(curvature) < 0:
        return

    rows, cols = img.shape[:2]
    
    # smooth curvature values
    curvature = signal.savgol_filter(curvature, window_length=3, polyorder=1, mode="wrap")

    n = len(curve)
    # finding seed point for floodfill
    max_pos = np.argmax(curvature)
    for i in range(1, n):
        if max_pos-i < 0 and max_pos-i < -n:
            return
        p1 = curve[max_pos-i]
        p2 = curve[(max_pos+i) % n]
        cx = int(sum(np.multiply([p1[0], p2[0], curve[max_pos][0]], [0.4, 0.4, 0.2])))
        cy = int(sum(np.multiply([p1[1], p2[1], curve[max_pos][1]], [0.4, 0.4, 0.2])))
        if cx >= 1 and cx < cols-1 and cy >= 1 and cy < rows-1:
            if curvature[max_pos-i] < 0 or curvature[(max_pos+i) % n] < 0:
                if not has_color_in_neighborhood(img, cx, cy, color):
                    cv2.floodFill(img, None, (cx, cy), color)
                    #draw_cross(img, cx, cy, (255, 255, 255))
                    return


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


def clear_rectangle_at_position(image, x, y, w, h, color):
    rows,cols = image.shape
    w1 = w // 2
    h1 = h // 2
    r1 = y - h1
    c1 = x - w1
    r = r1
    for i in range(0, h):
        c = c1
        for j in range(0, w):
            if r>=0 and r<rows and c>=0 and c<cols:
                image[r+i, c+j] = color


# move along the curve and clear curve pixels on image
def clear_curve(image, curve, w, h, color):
    if image is None:
        return
        
    if len(curve) == 0:
        return
    
    rows,cols = image.shape
    for p in curve:
        clear_rectangle_at_position(image, p[0], p[1], w, h, color)
