import numpy as np
import cv2
from math import sqrt

def dist_point_to_line(slope, slope_point, point):
    """
    Returns the distance from a point to a line given by
    its slope and a point that lies on the line.
    """

    sx, sy = slope_point
    px, py = point

    if np.isinf(slope):
        return abs(sx - px)
    else:
        a = slope
        b = -1
        c = (sy - (slope * sx))

        return abs(a * px + b * py + c) / sqrt(a**2 + b**2)

def dist_point_to_point(p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    return sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

def orthogonal_projection(slope, slope_point, point):
    """
    The orthogonal projection of a point is equal to the
    intersection between the projection line and the
    perpendicular line. 
    """

    perpendicular_slope = 0 if np.isinf(slope) else -1 / slope

    return intersection_lines(slope0=slope, point0=slope_point, slope1=perpendicular_slope, point1=point)

def mean_position(indices, face):
    """
    Calculates the mean position of all points included in the
    indices array of a given face.
    """

    x_avg = y_avg = 0
    size = len(indices)

    for index in indices:
        x_avg += face[index][0]
        y_avg += face[index][1]
    
    return (x_avg / size, y_avg / size)

def round_tuple(t):
    return tuple(map(lambda x: isinstance(x, float) and int(x) or x, t))

def intersection_lines(slope0, point0, slope1, point1):
    #https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

    a = b = None

    if np.isinf(slope0):
        a = np.array([[0, 1], [1, -slope1]])
        b = np.array([point0[0], point1[1] - slope1 * point1[0]])

    elif np.isinf(slope1):
        a = np.array([[1, -slope0], [0, 1]])
        b = np.array([point0[1] - slope0 * point0[0], point1[0]])

    else:
        a = np.array([[1, -slope0], [1, -slope1]])
        b = np.array([point0[1] - slope0 * point0[0], point1[1] - slope1 * point1[0]])

    y, x = np.linalg.solve(a, b)
    return (x, y)

def resize_with_aspectratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def ratio(x, y):
    return x / y if x < y else y / x

def normalize_uint8(arr):
    arr_cp = arr.copy()

    arr_cp -= arr_cp.min()
    arr_cp = (arr_cp / arr_cp.max()) * 255

    return arr_cp.astype(np.uint8)
