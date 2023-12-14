import math
from geopy.distance import geodesic


EARTH_RADIUS_KM = 6371.0

def euclidean_distance(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    lat_diff = math.radians(lat2 - lat1)
    lon_diff = math.radians(lon2 - lon1)
    distance = EARTH_RADIUS_KM * math.sqrt(lat_diff**2 + lon_diff**2)
    return distance

def manhattan_distance(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    lat_diff = math.radians(lat2 - lat1)
    lon_diff = math.radians(lon2 - lon1)
    distance = EARTH_RADIUS_KM * (abs(lat_diff) + abs(lon_diff))
    return distance

def geographic_distance(station1, station2):
    return geodesic(station1.position, station2.position).kilometers


def specify(actual_cost_estimate: str, heuristic_cost_estimate: str):
    if actual_cost_estimate == "euc":
        actual_cost = euclidean_distance
    elif actual_cost_estimate == "manh":
        actual_cost = manhattan_distance
    elif actual_cost_estimate == "geo":
        actual_cost = geographic_distance
    else:
        actual_cost = 0
    
    if heuristic_cost_estimate == "euc":
        heuristic_cost = euclidean_distance
    elif heuristic_cost_estimate == "manh":
        heuristic_cost = manhattan_distance
    elif heuristic_cost_estimate == "geo":
        heuristic_cost = geographic_distance
    else:
        heuristic_cost = 0
        
    return actual_cost, heuristic_cost