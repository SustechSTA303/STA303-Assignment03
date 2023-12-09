from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
from collections import deque
import math
import time
import argparse


# Implement the following function
def get_path_bfs(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
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

    queue = deque([start_station])
    visited = set()

    path = {start_station: None}

    while queue:
        current_station = queue.popleft()
        visited.add(current_station)

        if current_station == end_station:
            return reconstruct_path(path, end_station)

        for neighbor in current_station.links:
            if neighbor not in visited:
                queue.append(neighbor)
                path[neighbor] = current_station

    raise Exception("Path does not exist")


def get_path_Astar(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:

    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_list = [start_station]
    closed_list = set()

    path = {start_station: None}
    cost = {start_station: 0}
    
    while open_list:
        # current_station = min(open_list, key=lambda station: cost[station] + heuristic(station, end_station))
        # current_station = min(open_list, key=lambda station: cost[station] + heuristic_chebyshev(station, end_station))
        current_station = min(open_list, key=lambda station: cost[station] + heuristic_manhattan(station, end_station))
        open_list.remove(current_station)
        closed_list.add(current_station)

        if current_station == end_station:
            return reconstruct_path(path, end_station)

        for neighbor in current_station.links:
            if neighbor in closed_list:
                continue
            
            tentative_cost = cost[current_station] + distance(current_station, neighbor)
            if neighbor not in open_list or tentative_cost < cost[neighbor]:
                open_list.append(neighbor)
                path[neighbor] = current_station
                cost[neighbor] = tentative_cost
    
    raise Exception("Path does not exist")

def get_path_Bellman_Ford(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
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

    weight = {station: math.inf for station in map.values()}
    predecessor = {station: None for station in map.values()}
    weight[start_station] = 0

    for _ in range(len(map) - 1):
        for station in map.values():
            for neighbor in station.links:
                edge_weight = distance(station, neighbor)
                if weight[station] + edge_weight < weight[neighbor]:
                    weight[neighbor] = weight[station] + edge_weight
                    predecessor[neighbor] = station

    for station in map.values():
        for neighbor in station.links:
            edge_weight = distance(station, neighbor)
            if weight[station] + edge_weight < weight[neighbor]:
                raise Exception("Negative weight cycle detected. No shortest path exists.")
            
    path = []
    current_station = end_station
    while current_station:
        path.insert(0, current_station.name)
        current_station = predecessor[current_station]

    return path

def heuristic(station, end_station):
    return distance(station, end_station)

def heuristic_manhattan(station, end_station):
    return abs(station.position[0] - end_station.position[0]) + abs(station.position[1] - end_station.position[1])

def heuristic_chebyshev(station, end_station):
    return max(abs(station.position[0] - end_station.position[0]), abs(station.position[1] - end_station.position[1]))

def distance(station1, station2):
    return ((station1.position[0] - station2.position[0]) ** 2 + (station1.position[1] - station2.position[1]) ** 2) ** 0.5

def reconstruct_path(path, end_station):
    result = [end_station.name]
    while path[end_station]:
        end_station = path[end_station]
        result.append(end_station.name)
    return result[::-1]


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
    # path = get_path_Bellman_Ford(start_station_name, end_station_name, stations)
    path = get_path_bfs(start_station_name, end_station_name, stations)
    # path = get_path_Astar(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red

    start_time = time.time()
    plot_path(path, 'C:/Users/wyh/Desktop/STA303-Assignment03-main/STA303-Assignment03-main/visualization_underground/my_shortest_path_in_London_railway_BFS.html', stations, underground_lines)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")