from typing import List
from build_data import Station, Node

from .heuristic_function import build_heuristic_function


def bfs(start_station_name: str,
        end_station_name: str,
        map: dict[str, Station],
        heuristic_function: str) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    stations = {}
    for station_name in map:
        stations[station_name] = Node(station_name)
    stations[start_station_name].distance = 0
    queue = [(0, start_station_name)]
    heuristic_function = build_heuristic_function(heuristic_function)

    while len(queue) > 0:
        costs = {station_name: heuristic_function(station_name, end_station_name, map) for distance, station_name in
                 queue}
        best_station_name = min(costs, key=costs.get)
        queue = [distance_station for distance_station in queue if distance_station[1] != best_station_name]

        if stations[best_station_name].visited:
            continue
        stations[best_station_name].visited = True

        for neighbor in map[best_station_name].links:
            if stations[neighbor.name].visited:
                continue
            stations[neighbor.name].parent = stations[best_station_name]
            new_distance = stations[best_station_name].distance + map[best_station_name].edges[neighbor.name]
            stations[neighbor.name].distance = new_distance

            if neighbor.name == end_station_name:
                return stations[end_station_name].backward()

            queue.append((new_distance, neighbor.name))

    return stations[end_station_name].backward()
