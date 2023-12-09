from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import math
import argparse
from heapq import heappop, heappush
import time


def a_star_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[Station]:
    """
    Runs A* algorithm on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name
    Returns:
        List[Station]: A path composed of a series of Station objects
    """
    def distance(station1, station2):
        x1, y1 = station1.position
        x2, y2 = station2.position
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Heuristic function for A*
    def heuristic(node1, node2):
        return distance(node1, node2)

    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    # Priority queue for the open set
    open_set = [(0, start_station, [])]  # Updated: Include an empty list for the current path
    # Set to keep track of visited stations
    visited = set()
    
    while open_set:
        current_cost, current_station, current_path = heappop(open_set)

        if current_station == end_station:
            # Return the path when reaching the destination
            return current_path + [current_station.name]

        if current_station in visited:
            continue

        visited.add(current_station)

        for neighbor in current_station.links:
            if neighbor in visited:
                continue

            tentative_cost = current_cost + distance(current_station, neighbor)
            heuristic_cost = tentative_cost + distance(neighbor, end_station)
            heappush(open_set, (heuristic_cost, neighbor, current_path + [current_station.name]))
            
    return []


def a_star_euclidean_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[Station]:

    def a_star_euclidean_heuristic(node1, node2):
        x1, y1 = node1.position
        x2, y2 = node2.position
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  
    def heuristic(node1, node2):
        return a_star_euclidean_heuristic(node1, node2)

        # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    # Priority queue for the open set
    open_set = [(0, start_station, [])]  # Updated: Include an empty list for the current path
    # Set to keep track of visited stations
    visited = set()
    
    while open_set:
        current_cost, current_station, current_path = heappop(open_set)

        if current_station == end_station:
            # Return the path when reaching the destination
            return current_path + [current_station.name]

        if current_station in visited:
            continue

        visited.add(current_station)

        for neighbor in current_station.links:
            if neighbor in visited:
                continue

            tentative_cost = current_cost + distance(current_station, neighbor)
            heuristic_cost = tentative_cost + distance(neighbor, end_station)
            heappush(open_set, (heuristic_cost, neighbor, current_path + [current_station.name]))
            
    return []


def a_star_manhattan_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[Station]:
    
    def a_star__heuristic(node1, node2):
        x1, y1 = node1.position
        x2, y2 = node2.position
        return abs(x2 - x1) + abs(y2 - y1)
    
    def heuristic(node1, node2):
        return a_star_manhattan_heuristic(node1, node2)

    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    # Priority queue for the open set
    open_set = [(0, start_station, [])]  # Updated: Include an empty list for the current path
    # Set to keep track of visited stations
    visited = set()
    
    while open_set:
        current_cost, current_station, current_path = heappop(open_set)

        if current_station == end_station:
            # Return the path when reaching the destination
            return current_path + [current_station.name]

        if current_station in visited:
            continue

        visited.add(current_station)

        for neighbor in current_station.links:
            if neighbor in visited:
                continue

            tentative_cost = current_cost + distance(current_station, neighbor)
            heuristic_cost = tentative_cost + distance(neighbor, end_station)
            heappush(open_set, (heuristic_cost, neighbor, current_path + [current_station.name]))
            
    return []



def a_star_custom_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[Station]:
    
    def a_star__heuristic(node1, node2):
    # Custom heuristic: Number of links from the current node to the goal node
        return len(node2.links)
    
    def heuristic(node1, node2):
        return a_star_custom_heuristic(node1, node2)

        # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    # Priority queue for the open set
    open_set = [(0, start_station, [])]  # Updated: Include an empty list for the current path
    # Set to keep track of visited stations
    visited = set()
    
    while open_set:
        current_cost, current_station, current_path = heappop(open_set)

        if current_station == end_station:
            # Return the path when reaching the destination
            return current_path + [current_station.name]

        if current_station in visited:
            continue

        visited.add(current_station)

        for neighbor in current_station.links:
            if neighbor in visited:
                continue

            tentative_cost = current_cost + distance(current_station, neighbor)
            heuristic_cost = tentative_cost + distance(neighbor, end_station)
            heappush(open_set, (heuristic_cost, neighbor, current_path + [current_station.name]))
            
    return []


def calculate_path_length(path, map):
    # Assuming the path is a list of station names
    length = 0
    for i in range(len(path)-1):
        length += distance(map[path[i]], map[path[i+1]])
    return length


def distance(station1, station2):
    x1, y1 = station1.position
    x2, y2 = station2.position
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


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
    stations_map, underground_lines = build_data()

    # Run euclidean_heuristic algorithm
    start_time_euclidean_heuristic = time.time()
    path_euclidean_heuristic = a_star_euclidean_get_path(start_station_name, end_station_name, stations_map)
    elapsed_time_euclidean_heuristic = time.time() - start_time_euclidean_heuristic
    path_length_euclidean_heuristic = calculate_path_length(path_euclidean_heuristic, stations_map)

    # Run manhattan_heuristic algorithm
    start_time_manhattan_heuristic = time.time()
    path_manhattan_heuristic = a_star_manhattan_get_path(start_station_name, end_station_name, stations_map)
    elapsed_time_manhattan_heuristic = time.time() - start_time_manhattan_heuristic
    path_length_manhattan_heuristic = calculate_path_length(path_manhattan_heuristic, stations_map)

    # Run custom_heuristic algorithm
    start_time_custom_heuristic = time.time()
    path_custom_heuristic = a_star_custom_get_path(start_station_name, end_station_name, stations_map)
    elapsed_time_custom_heuristic = time.time() - start_time_custom_heuristic
    path_length_custom_heuristic = calculate_path_length(path_custom_heuristic, stations_map)
    
    # Compare results
    print(f"euclidean_heuristic - Elapsed Time: {elapsed_time_euclidean_heuristic}, Path Length: {path_length_euclidean_heuristic}")
    print(f"manhattan_heuristic - Elapsed Time: {elapsed_time_manhattan_heuristic}, Path Length: {path_length_manhattan_heuristic}")
    print(f"custom_heuristic - Elapsed Time: {elapsed_time_custom_heuristic}, Path Length: {path_length_custom_heuristic}")
    
    # Compare path lengths
    if path_length_euclidean_heuristic < path_length_manhattan_heuristic and path_length_euclidean_heuristic < path_length_custom_heuristic:
        print("euclidean_heuristic found the shortest path.")
    elif path_length_manhattan_heuristic < path_length_euclidean_heuristic and path_length_manhattan_heuristic < path_length_custom_heuristic:
        print("manhattan_heuristic found the shortest path.")
    elif path_length_custom_heuristic < path_length_euclidean_heuristic and path_length_custom_heuristic < path_length_manhattan_heuristic:
        print("custom_heuristic found the shortest path.")
    # Compare path lengths to find the longest path
    elif path_length_euclidean_heuristic > path_length_manhattan_heuristic and path_length_euclidean_heuristic > path_length_custom_heuristic:
        print("euclidean_heuristic found the longest path.")
    elif path_length_manhattan_heuristic > path_length_euclidean_heuristic and path_length_manhattan_heuristic > path_length_custom_heuristic:
         print("manhattan_heuristic found the longest path.")
    elif path_length_custom_heuristic > path_length_euclidean_heuristic and path_length_custom_heuristic > path_length_manhattan_heuristic:
         print("custom_heuristic found the longest path.")
    else:
        print("All algorithms found paths with the same length.")
