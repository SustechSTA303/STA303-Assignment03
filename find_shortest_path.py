from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from Algorithm import astar_algorithm,spfa_algorithm, dijkstra_algorithm, bellman_ford_algorithm, ucs_algorithm,haversine_distance,manhattan_distance,euclidean_distance，chebyshev_distance
from queue import PriorityQueue
import time

# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], algorithm: str) -> List[str]:
    """
    runs astar on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): str The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    if algorithm == 'astar':
        path, total_distance = astar_algorithm(start_station_name, end_station_name, map, distance_function='haversine')
    elif algorithm == 'dijkstra':
        path, total_distance = dijkstra_algorithm(start_station_name, end_station_name,map)
    elif algorithm == 'bellman_ford':
        path, total_distance = bellman_ford_algorithm(start_station_name, end_station_name,map)
    elif algorithm == 'ucs':
        path, total_distance = ucs_algorithm(start_station_name, end_station_name,map)
    elif algorithm == 'spfa':
        path, total_distance = spfa_algorithm(start_station_name, end_station_name, map)
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
#     print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
#     print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    formatted_path = ' -> '.join(path)
    print(f"Path: {formatted_path}")
    print(f"Total Distance: {total_distance} km")
#     print(f"Expand Notes："{expanded_nodes})
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('--algorithm', type=str, default='astar', choices=['astar','spfa','dijkstra','bellman_ford','ucs'])
    parser.add_argument('--distance', type=str, default='haversine', choices=['haversine', 'manhattan', 'euclidean','chebyshev'])
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    algorithm = args.algorithm
    distance_function = args.distance

    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations, algorithm)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)