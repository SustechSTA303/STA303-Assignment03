from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue
import math


def distance(station1, station2):
    # 计算两个站点之间的直线距离（欧几里得距离）
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    distance_km = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return distance_km
def get_path_dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    Runs Dijkstra's algorithm on the map, find the shortest path between start and end stations.
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[str]: A path composed of a series of station names
    """
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Priority queue to store nodes with their current costs
    open_set = PriorityQueue()
    open_set.put((0, start_station, [start_station]))

    # Dictionary to store the cost to reach each station
    cost_to_reach = {start_station: 0}

    while not open_set.empty():
        current_cost, current_station, current_path = open_set.get()

        if current_station == end_station:
            # Convert the path to a list of station names
            path_names = [station.name for station in current_path]
            return path_names  # Found the path

        for neighbor in current_station.links:
            new_cost = current_cost + distance(current_station, neighbor)

            # If the neighbor has not been visited or a shorter path is found
            if neighbor not in cost_to_reach or new_cost < cost_to_reach[neighbor]:
                cost_to_reach[neighbor] = new_cost
                new_path = current_path + [neighbor]
                open_set.put((new_cost, neighbor, new_path))

    return None  # No path found

# The distance and estimate_heuristic functions remain the same as in the A* implementation

if __name__ == '__main__':
    # The rest of the script remains unchanged
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()
    path = get_path_dijkstra(start_station_name, end_station_name, stations)
    print(f"Path: {path}")

    if path is not None:
        plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway_dijkstra.html', stations, underground_lines)
    else:
        print("Unable to find a path.")
