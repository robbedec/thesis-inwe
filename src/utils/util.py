from math import sqrt

def dist_point_to_line(slope, slope_point, point):
    """
    Returns the distance from a point to a line given by
    the its slope and a point that lies on the line.
    """

    sx, sy = slope_point
    px, py = point

    a = 1
    b = -slope
    c = (slope * sx) - sy

    return abs(a * px + b * py + c) / sqrt(a**2 + b**2)

def dist_point_to_point(p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    return sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

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