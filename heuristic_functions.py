from build_data import Station, build_data


# The absolute difference in stops
def heuristic1(node: str, goal: str) -> float:
    return abs(len(stations[node].links) - len(stations[goal].links))


# Euclidean distance
def heuristic2(node: str, goal: str) -> float:
    pos1 = stations[node].position
    pos2 = stations[goal].position
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    
# Manhattan distance
def heuristic3(node: str, goal: str) -> float:
    pos1 = stations[node].position
    pos2 = stations[goal].position
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
