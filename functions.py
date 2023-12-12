from math import sqrt
import random
from geopy.distance import geodesic
from build_data import Station

# * heuristics function
def heuristics(station1: Station, station2: Station, type: str, weight: float = 1.0) -> float:
    if type == "Haversine":
        return weight * geodesic(station1.position, station2.position).kilometers
    # elif type == "Manhattan":
    #     return weight * abs(station1.position[0] - station2.position[0]) + abs(
    #         station1.position[1] - station2.position[1]
    #     )
    elif type == "Euclidean":
        return weight * sqrt(
            (station1.position[0] - station2.position[0]) ** 2
            + (station1.position[1] - station2.position[1]) ** 2
        )

# * cost function
def cost(station1: Station, station2: Station, type: str) -> float:
    if type == "1":
        return 1.0
    if type == "Haversine":
        return geodesic(station1.position, station2.position).kilometers
    # elif type == "Manhattan":
    #     return abs(station1.position[0] - station2.position[0]) + abs(
    #         station1.position[1] - station2.position[1]
    #     )
    elif type == "Euclidean":
        return sqrt(
            (station1.position[0] - station2.position[0]) ** 2
            + (station1.position[1] - station2.position[1]) ** 2
        )

#* path length function
def pathLength(path: list,map: dict[str, Station]) -> float:
    total_length = 0
    for i in range(len(path) - 1):
        total_length += cost(map[path[i]], map[path[i + 1]], "Haversine")
    return total_length

# * given a number `n`, randomly choose n pairs of station
def random_choice(station_name: list, number :int) -> list:
    all_pairs = [(a, b) for a in station_name for b in station_name if a != b]
    chosen_pairs = random.sample(all_pairs, number)
    return chosen_pairs