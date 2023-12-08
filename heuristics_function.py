from math import sqrt

from build_data import Station

#* heuristics function
def Euclidean(station1: Station, station2: Station) -> float:
    """
    Euclidean distance heuristic for A*.
    """
    return sqrt((station1.position[0] - station2.position[0]) ** 2 +
                (station1.position[1] - station2.position[1]) ** 2)