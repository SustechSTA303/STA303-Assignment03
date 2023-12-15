from math import log
from build_data import Station, build_data


stations, underground_lines = build_data()



# Euclidean distance
def euclidean_distance(node: str, goal: str) -> float:
    pos1 = stations[node].position
    pos2 = stations[goal].position
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    
# Manhattan distance
def manhattan_distance(node: str, goal: str) -> float:
    pos1 = stations[node].position
    pos2 = stations[goal].position
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Chebyshev_distance
def chebyshev_distance(node: str, goal: str) -> float:
    pos1 = stations[node].position
    pos2 = stations[goal].position
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

# Minkowski_distance
def minkowski_distance(node: str, goal: str, p: float) -> float:
    pos1 = stations[node].position
    pos2 = stations[goal].position
    return ((abs(pos1[0] - pos2[0])**p + abs(pos1[1] - pos2[1])**p)**(1/p))


