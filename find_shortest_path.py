from queue import PriorityQueue
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from collections import deque
import heapq
from sys import maxsize
import time
from math import radians, sin, cos, sqrt, atan2

# Implement the following function
class Node:
    def __init__(self, station, cost, heuristic, parent=None):
        self.station = station
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent  # Initialize the parent attribute

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def Haversine(position1, position2):
    # Define your heuristic function here (e.g., Euclidean distance)
    lat1, lon1 = position1
    lat2, lon2 = position2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # Haversine 公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # 地球半径（单位：km）
    R = 6371.0
    distance = R * c
    return distance

def h_manhattan(position1, position2):
    lat1, lon1 = position1
    lat2, lon2 = position2
    return abs(lat2 - lat1) + abs(lon2 - lon1)

def h_euclidean(position1, position2):
    lat1, lon1 = position1
    lat2, lon2 = position2
    return ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5

def chebyshev_distance(position1, position2):
    lat1, lon1 = position1
    lat2, lon2 = position2
    return max(abs(lat2 - lat1), abs(lon2 - lon1))

def custom_heuristic(position1, position2):
    lat1, lon1 = position1
    lat2, lon2 = position2
    euclidean_dist = ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5
    manhattan_dist = abs(lat2 - lat1) + abs(lon2 - lon1)
    return 0.1 * euclidean_dist + 0.9 * manhattan_dist


def calculate_cost(current_node, neighbor):
    return current_node.cost + Haversine(current_node.station.position, neighbor.position)

def astar(start_station, end_station, map, heuristic):
    open_set = PriorityQueue()
    closed_set = set()

    start_node = Node(start_station, 0, 0)
    open_set.put(start_node)

    while not open_set.empty():
        current_node = open_set.get()

        if current_node.station == end_station:
            # Reconstruct the path
            path = []
            while current_node:
                path.append(current_node.station.name)
                current_node = current_node.parent
            return path[::-1]

        if current_node.station in closed_set:
            continue

        closed_set.add(current_node.station)

        for neighbor in current_node.station.links:
            if neighbor not in closed_set:
                cost = calculate_cost(current_node, neighbor)
                heuristic_val = heuristic(neighbor.position, end_station.position)
                new_node = Node(neighbor, cost, heuristic_val + cost)
                new_node.parent = current_node
                open_set.put(new_node)

    return None  # No path found

#BFS
def bfs(start_station, end_station, map):
    queue = deque([(start_station, [start_station.name])])
    visited = set()

    while queue:
        current_station, path = queue.popleft()
        visited.add(current_station)

        if current_station == end_station:
            return path

        for neighbor in current_station.links:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor.name]))
                visited.add(neighbor)

    return None

def dijkstra(start_station, end_station, map):
    open_set = PriorityQueue()
    closed_set = set()

    start_node = Node(start_station, 0, 0)
    open_set.put(start_node)

    while not open_set.empty():
        current_node = open_set.get()

        if current_node.station == end_station:
            # Reconstruct the path
            path = []
            while current_node:
                path.append(current_node.station.name)
                current_node = current_node.parent
            return path[::-1]

        if current_node.station in closed_set:
            continue

        closed_set.add(current_node.station)

        for neighbor in current_node.station.links:
            if neighbor not in closed_set:
                cost = calculate_cost(current_node, neighbor)
                heuristic_val = 0
                new_node = Node(neighbor, cost, heuristic_val + cost)
                new_node.parent = current_node
                open_set.put(new_node)

    return None  # No path found

def bellman_ford(start_station, end_station, map):
    distances = {station: maxsize for station in map.values()}
    distances[start_station] = 0
    predecessor = {station: None for station in map.values()}

    for _ in range(len(map) - 1):
        for current_station in map.values():
            for neighbor in current_station.links:
                if distances[current_station] + 1 < distances[neighbor]:
                    distances[neighbor] = distances[current_station] + 1
                    predecessor[neighbor] = current_station

    # Check for negative cycles
    for current_station in map.values():
        for neighbor in current_station.links:
            if distances[current_station] + 1 < distances[neighbor]:
                print("Negative cycle detected. Bellman-Ford algorithm cannot handle negative cycles.")
                return None

    # Reconstruct the path
    path = [end_station.name]
    while predecessor[end_station]:
        path.insert(0, predecessor[end_station].name)
        end_station = predecessor[end_station]

    return path

def greedy_best_first_search(start_station, end_station, map):
    open_set = PriorityQueue()
    closed_set = set()

    start_node = Node(start_station, 0, 0)
    open_set.put(start_node)

    while not open_set.empty():
        current_node = open_set.get()

        if current_node.station == end_station:
            # Reconstruct the path
            path = []
            while current_node:
                path.append(current_node.station.name)
                current_node = current_node.parent
            return path[::-1]

        if current_node.station in closed_set:
            continue

        closed_set.add(current_node.station)

        for neighbor in current_node.station.links:
            if neighbor not in closed_set:
                heuristic_val = Haversine(neighbor.position,
                                          end_station.position)  # You need to define a heuristic function h
                new_node = Node(neighbor, 0, heuristic_val)
                new_node.parent = current_node
                open_set.put(new_node)

    return None  # No path found
def calculate_path_length(path, map):
    length = 0
    for i in range(len(path) - 1):
        current_station = map[path[i]]
        next_station = map[path[i + 1]]
        length += Haversine(current_station.position, next_station.position)
    return length

def get_path_and_time(algorithm, start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_function=None) -> List[str]:
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
    start_time = time.time()
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    #print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    #print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    #pass
    if heuristic_function:
        path = algorithm(start_station, end_station, stations, heuristic_function)
        if path:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"a star algorithm with {heuristic_function.__name__} took {elapsed_time:.6f} seconds.")
            print(f"a star algorithm with {heuristic_function.__name__} Path Length: {calculate_path_length(path, map):.2f} units.")
            return path
        else:
            print("No path found.")
            return []
    else:
        path = algorithm(start_station, end_station, stations)
        if path:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"{algorithm.__name__} Algorithm took {elapsed_time:.6f} seconds.")
            print(f"{algorithm.__name__} Path Length: {calculate_path_length(path, map):.2f} units.")
            return path
        else:
            print("No path found.")
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
    # A* algorithm
    heuristic_functions = [
        ('Haversine', Haversine),
        ('Manhattan', h_manhattan),
        ('Euclidean', h_euclidean),
        ('Chebyshev Distance', chebyshev_distance),
        ('weighted_heuristic', custom_heuristic)
    ]

    for heuristic_name, heuristic_function in heuristic_functions:
        path = get_path_and_time(astar, start_station_name, end_station_name, stations, heuristic_function)
        plot_path(path, f'visualization_underground/my_path_in_London_railway_astar_{heuristic_name}.html', stations,
                  underground_lines)


    algorithms = [bfs, dijkstra, bellman_ford, greedy_best_first_search]

    for algorithm in algorithms:
        path = get_path_and_time(algorithm, start_station_name, end_station_name, stations)
        plot_path(path, f'visualization_underground/my_path_in_London_railway_{algorithm.__name__}.html', stations,
                  underground_lines)
