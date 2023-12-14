import math


def Euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def Manhattan_distance(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def Diagonal_distance(x1, x2, y1, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)