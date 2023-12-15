from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq

class Node:
    def __init__(self, station, distance, parent=None):
        self.station = station
        self.distance = distance
        self.parent = parent

    def __lt__(self, other):
        return self.distance < other.distance

def dijkstra(start_station, end_station, map):
    priority_queue = []
    heapq.heappush(priority_queue, Node(start_station, 0))

    visited = set()

    while priority_queue:
        current_node = heapq.heappop(priority_queue)

        if current_node.station == end_station:
            path = []
            while current_node:
                path.insert(0, current_node.station.name)
                current_node = current_node.parent
            return path

        if current_node.station in visited:
            continue

        visited.add(current_node.station)

        for neighbor_station in current_node.station.links:
            if neighbor_station in visited:
                continue

            tentative_distance = current_node.distance + 1  # Assuming equal weight for all edges

            neighbor_node = Node(
                station=neighbor_station,
                distance=tentative_distance,
                parent=current_node
            )

            heapq.heappush(priority_queue, neighbor_node)

    return None

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    path = dijkstra(start_station, end_station, map)

    if path:
        print("Shortest path found:", path)
        return path
    else:
        return ["No path found"]

if __name__ == '__main__':
    # Rest of the code remains unchanged
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
