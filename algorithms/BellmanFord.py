from typing import List
from build_data import Station, Node


def bellman_ford(start_station_name: str,
                 end_station_name: str,
                 map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    stations = {}
    for station_name in map:
        stations[station_name] = Node(station_name)
    stations[start_station_name].distance = 0
    queue = [(0, start_station_name)]

    while len(queue) > 0:
        distance, station_name = queue.pop(0)

        if stations[station_name].visited:
            continue
        stations[station_name].visited = True

        for neighbor in map[station_name].links:
            new_distance = distance + map[station_name].edges[neighbor.name]
            queue.append((new_distance, neighbor.name))
            if new_distance < stations[neighbor.name].distance:
                stations[neighbor.name].distance = new_distance
                stations[neighbor.name].parent = stations[station_name]
                queue.append((new_distance, neighbor.name))

    return stations[end_station_name].backward()
