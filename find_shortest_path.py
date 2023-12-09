from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from heuristic_functions import heuristic2
import heapq


# Implement the following function

def get_path(start_station_name: str, end_station_name: str, stations_map: dict[str, Station]) -> List[str]:
    """
    Runs A* on the map, find the shortest path between start and end stations.
    Args:
        start_station_name (str): The name of the starting station.
        end_station_name (str): The name of the ending station.
        stations_map (Dict[str, Station]): Mapping between station names and station objects.
    Returns:
        List[str]: A list of station names representing the shortest path.
    """
    start_station = stations_map[start_station_name]
    end_station = stations_map[end_station_name]

    # Priority queue to store the open set of nodes (ordered by total cost)
    open_set = [(0, start_station, [])]

    # Set to keep track of visited nodes
    visited = set()

    while open_set:
        current_cost, current_station, current_path = heapq.heappop(open_set)

        if current_station == end_station:
            return current_path + [current_station.name]

        if current_station in visited:
            continue

        visited.add(current_station)

        for neighbor_name in current_station.links:
            neighbor = stations_map[neighbor_name]
            distance = 1  # You might need to replace this with the actual distance calculation based on coordinates

            if neighbor not in visited:
                # Calculate the total cost for the neighbor
                total_cost = current_cost + distance + heuristic2(neighbor.name, end_station.name)

                # Push the neighbor onto the open set with the total cost and updated path
                heapq.heappush(open_set, (total_cost, neighbor, current_path + [current_station.name]))

    return []



if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
