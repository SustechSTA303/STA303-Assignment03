from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
from queue import PriorityQueue
from collections import deque
from sys import maxsize
import argparse
import heapq
import time

class Node:
    def __init__(self, station, cost, heuristic, parent=None):
        self.station = station
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent  # Initialize the parent attribute

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)
    
def manhattan(a, b):  
    lat_a, lon_a = a
    lat_b, lon_b = b
    return abs(lat_a - lat_b) + abs(lon_a - lon_b)

def euclidean(a, b):
    lat_a, lon_a = a
    lat_b, lon_b = b
    return ((lat_a - lat_b)**2 + (lon_a - lon_b)**2)**0.5

def chebyshev(a, b):
    lat_a, lon_a = a
    lat_b, lon_b = b
    return max(abs(lat_a - lat_b), abs(lon_a - lon_b))

# should change heuristic function here
def calculate_cost(current_node, neighbor):
    return current_node.cost + manhattan(current_node.station.position, neighbor.position)


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

# No need for heuristic function
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

def dijkstra(start_station, end_station, map, heuristic):
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

def greedy_best_first_search(start_station, end_station, map, heuristic):
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
                heuristic_val = manhattan(neighbor.position,
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
        
        # Using different heuristic function that defined above, can change it into Mahattan, Euclidean or Chebyshev distance.
        length += manhattan(current_station.position, next_station.position)
        
    return length


# heuristic_function=None使得就算没有启发式距离函数也可以完成下列算法
def get_path(algorithm, start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_function=None) -> List[str]:
    """
    runs astar on the map, find the shortest path between a and b
    Args:
@@ -18,13 +226,35 @@ def get_path(start_station_name: str, end_station_name: str, map: dict[str, Stat
        List[Station]: A path composed of a series of station_name
    """
    start_time = time.time()
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    # print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    # print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    # pass

    if heuristic_function:
        if algorithm == 'bfs':
            path = bfs(start_station, end_station, stations)
        else:
            path = algorithm(start_station, end_station, stations, heuristic_function)
        
        if path:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Using {algorithm.__name__} with {heuristic_function.__name__} distance took {elapsed_time:.6f} seconds.")
            print(f"Using {algorithm.__name__} with {heuristic_function.__name__} distance, the path length is {calculate_path_length(path, map):.2f} units.")
            return path
        else:
            print("No path found.")
            return []
    else:
        path = algorithm(start_station, end_station, stations)
        if path:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Using {algorithm.__name__} took {elapsed_time:.6f} seconds.")
            print(f"Using {algorithm.__name__}, the path length is {calculate_path_length(path, map):.2f} units.")
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
    
#     # Using different heuristic function in the same algorithm
#     heuristic_function = [('Manhattan', manhattan), ('Euclidean', euclidean),('Chebyshev Distance', chebyshev)]

#     # Changing algorithm below into those defined above in get_path
#     for heuristic_name, heuristic_function in heuristic_function:
#         path = get_path(astar, start_station_name, end_station_name, stations, heuristic_function)
#         plot_path(path, f'visualization_underground/my_path_in_London_railway_astar_{heuristic_name}.html', stations,
#                   underground_lines)

    # Should change heuristic function here
    
    path_astar = get_path(astar, start_station_name, end_station_name, stations, manhattan)
    plot_path(path_astar, f'visualization_underground/my_path_in_London_railway_astar.html', stations,
                  underground_lines)
    path_bfs = get_path(bfs, start_station_name, end_station_name, stations)
    plot_path(path_bfs, f'visualization_underground/my_path_in_London_railway_bfs.html', stations,
                  underground_lines)
    path_dijkstra = get_path(dijkstra, start_station_name, end_station_name, stations, manhattan)
    plot_path(path_dijkstra, f'visualization_underground/my_path_in_London_railway_dijkstra.html', stations,
                  underground_lines)
    path_gbfs = get_path(greedy_best_first_search, start_station_name, end_station_name, stations, manhattan)
    plot_path(path_gbfs, f'visualization_underground/my_path_in_London_railway_greedy_best_first_search.html', stations,
                  underground_lines)

    