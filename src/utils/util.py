import numpy as np
import cv2
import operator

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
    """
    Normalizes array values to the interval [0, 255].
    Useful for images (uint8 interval). 
    """

    arr_cp = arr.copy()

    arr_cp -= arr_cp.min()
    arr_cp = (arr_cp / arr_cp.max()) * 255

    return arr_cp.astype(np.uint8)

def ROI_points_linear(points, padding=(20, 20), horizontal=True):
    """
    Creates a RotatedRectangle the respresents a Region Of Interest
    for given 2D points (approx colinear!).

    The points are fitted around a line and the rectangle is built
    around this line. Uses parametric equation of a line.

    - points: array of 2D points
    - padding: tuple containing the padding in the x and y direction
               applied according to the line fitted through the points.
    - horizontal: indication of the point alignment, are points more
                  vertically aligned than horizontally.

    Return the corners of the ROI
    """

    line = cv2.fitLine(points=points, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)

    vx, vy, x0, y0 = line.reshape((1, 4))[0]

    # Take the smallest x-coord and calculate parameter t
    # Calculate the y-coord using t, this point lies on the line.
    if horizontal:
        x_bottom = points[:,0].min()
        t = (x_bottom - x0) / vx
        y_bottom = y0 + t * vy

        x_top = points[:,0].max()
        t = (x_top - x0) / vx
        y_top = y0 + t * vy
    
    else:
        y_bottom = points[:,1].min()
        t = (y_bottom - y0) / vy
        x_bottom = x0 + t * vx

        y_top = points[:,1].max()
        t = (y_top - y0) / vy
        x_top = x0 + t * vx

    p1 = (x_bottom, y_bottom)
    p2 = (x_top, y_top)

    # ROI is a 3-tuple that contains the center tuple, size tuple and an angle.
    roi = cv2.minAreaRect(np.array([p1, p2], dtype='float32'))
    # Change increase rect width and height to create a region of interest
    roi = (roi[0], tuple(map(operator.add, roi[1], padding)), roi[2])

    # Get corners of ROI and convert to int
    box = cv2.boxPoints(roi)
    box = np.int0(box)

    return box