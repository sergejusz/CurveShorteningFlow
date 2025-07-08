import numpy as np


# curve is always represented by np.array([x1,x2,...], [y1,y2,...]])
# curve.shape = (2,n)

def get_curve_size(curve):
    return curve.shape[1:][0]


def is_empty_curve(curve):
    return curve.size == 0
    

def get_empty_curve():
        return np.array([[], []])
