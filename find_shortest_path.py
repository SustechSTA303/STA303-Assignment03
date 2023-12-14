from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time

import search_algorithm
import distance

# Implement the following function
def get_path(start_station_name: str, end_station_name: str, algorithm: str, distance_1:str, distance_2:str, map: dict[str, Station]) -> List[str]:
    """
    runs astar on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        algorithm(str): The name of the search algorithm used
        distance_1(distance function): Actual cost
        distance_2(distance function): Heuristic cost
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    # Measure execution time
    start_time = time.time()
    
    if algorithm == "Astar":
        path = search_algorithm.a_star(start_station, end_station, distance_1, distance_2, map)
    elif algorithm == "Ucs":
        path = search_algorithm.uniform_cost(start_station, end_station, distance_1, map)
    elif algorithm == "GreedyBFS":
        path = search_algorithm.greedy_bfs(start_station, end_station, distance_2, map)
    
    # Measure total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    #calculate the distance of the path
    total_cost = 0.0
    for i in range(len(path) - 1):
        current_station = map[path[i]]
        next_station = map[path[i + 1]]
        total_cost += distance.specify(distance_1,0)[0](current_station, next_station)
    
    # Print output
    
    print(f"Algorithm: {algorithm}")
    print(f"Actual Cost Function: {distance_1}")
    print(f"Heuristic Cost Function: {distance_2}")
    print(f"Time: {execution_time} seconds")
    print(f"Path Length: {total_cost}")
    
    
    return path, execution_time, total_cost
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    #print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    #print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    #pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('algorithm', type=str, help='algorithm')
    parser.add_argument('dis_1', type=str, help='dis_1')
    parser.add_argument('dis_2', type=str, help='dis_2')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    algorithm = args.algorithm
    dis_1 = args.dis_1
    dis_2 = args.dis_2

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path, execution_time, total_cost = get_path(start_station_name, end_station_name, algorithm, dis_1, dis_2, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
