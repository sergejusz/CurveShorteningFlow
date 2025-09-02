import numpy as np


# curve is always represented by np.array([x1,x2,...], [y1,y2,...]])
# curve.shape = (2,n)

def get_curve_size(curve):
    return curve.shape[1:][0]


def is_empty_curve(curve):
    return curve.size == 0
    

def get_empty_curve():
        return np.array([[], []])


def get_ellipse(cx, cy, radiusx, radiusy, num_points):
    t = np.linspace(0.0, 2.0*np.pi, num_points)
    return np.array([cx + radiusx*np.cos(t), cy + radiusy*np.sin(t)])


def get_circle(cx, cy, radius, num_points):
    return get_ellipse(cx, cy, radius, radius, num_points)

def get_paperclip(cx, cy, radius, num_points):
    n1 = int((np.pi * num_points)/(3 + np.pi))
    n2 = num_points - n1
    t1 = np.linspace(1.5 * np.pi, 0.5 * np.pi, n1 // 2, endpoint=False)
    curve = np.array([radius * np.cos(t1), radius * np.sin(t1)])
    print("First sin=", curve[0][0], curve[1][0], " last =", curve[0][-1], curve[1][-1])
    z1 = np.array([np.linspace(0.0, 3.0*radius, n2 // 2, endpoint=False), np.full(n2 // 2, radius)])
    print("First z1 = ", z1[0][0], z1[1][0], " last z1=", z1[0][-1], z1[1][-1])
    t2 = np.linspace(2.5 * np.pi, 1.5 * np.pi, n1 - ( n1 // 2), endpoint=False)
    z2 = np.array([3.0 * radius + radius * np.cos(t2), radius * np.sin(t2)])
    print("First z2 = ", z2[0][0], z2[1][0], " last z2=", z2[0][-1], z2[1][-1])
    z3 = np.array([np.linspace(3.0*radius, 0.0, n2 - (n2 // 2), endpoint=False), np.full(n2 - (n2 // 2), -radius)])
    print("First z3 = ", z3[0][0], z3[1][0], " last z3=", z3[0][-1], z3[1][-1])
    curve = np.append(curve, z1, axis=1)
    curve = np.append(curve, z2, axis=1)
    curve = np.append(curve, z3, axis=1)
    return np.add(curve, [[cx], [cy]])


