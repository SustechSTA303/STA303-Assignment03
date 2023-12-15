from typing import List
from build_data import Station, Node

def dfs(start_station_name: str,
        end_station_name: str,
        map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    stations = {}
    for station_name in map:
        stations[station_name] = Node(station_name)
    stations[start_station_name].distance = 0
    queue = [(0, start_station_name)]

    shortest_path = []
    shortest_path_distance = float("inf")

    while len(queue) > 0:
        distance, station_name = queue.pop(len(queue) - 1)

        if stations[station_name].visited:
            continue
        stations[station_name].visited = True

        for neighbor in map[station_name].links:
            if stations[neighbor.name].visited:
                continue
            new_distance = distance + map[station_name].edges[neighbor.name]
            stations[neighbor.name].distance = new_distance
            stations[neighbor.name].parent = stations[station_name]

            if neighbor.name == end_station_name:
                return stations[end_station_name].backward()
            queue.append((new_distance, neighbor.name))
    return stations[end_station_name].backward()
