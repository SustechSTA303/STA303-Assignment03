from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time
import pandas as pd
import argparse
from typing import Dict, List
from heapq import heappop, heappush
import requests
from math import radians, sin, cos, sqrt, atan2
# Implement the following function
# def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
#     """
#     runs astar on the map, find the shortest path between a and b
#     Args:
#         start_station_name(str): The name of the starting station
#         end_station_name(str): str The name of the ending station
#         map(dict[str, Station]): Mapping between station names and station objects of the name,
#                                  Please refer to the relevant comments in the build_data.py
#                                  for the description of the Station class
#     Returns:
#         List[Station]: A path composed of a series of station_name
#     """
#     # You can obtain the Station objects of the starting and ending station through the following code
#     start_station = map[start_station_name]
#     end_station = map[end_station_name]
#     # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
#     print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
#     print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
#     pass
# if __name__ == '__main__':

#     # 创建ArgumentParser对象
#     parser = argparse.ArgumentParser()
#     # 添加命令行参数
#     parser.add_argument('start_station_name', type=str, help='start_station_name')
#     parser.add_argument('end_station_name', type=str, help='end_station_name')
#     args = parser.parse_args()
#     start_station_name = args.start_station_name
#     end_station_name = args.end_station_name

#     # The relevant descriptions of stations and underground_lines can be found in the build_data.py
#     stations, underground_lines = build_data()
#     path = get_path(start_station_name, end_station_name, stations)
#     # visualization the path
#     # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
#     plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)

# A*欧式距离
def get_path1(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
#     """
#     Runs A* on the map to find the shortest path between start_station_name and end_station_name.
#     Args:
#         start_station_name (str): The name of the starting station
#         end_station_name (str): The name of the ending station
#         map (Dict[str, Station]): Mapping between station names and station objects of the name,
#                                   Please refer to the relevant comments in the build_data.py
#                                   for the description of the Station class
#     Returns:
#         List[str]: A list of station names representing the shortest path
#     """
#     # Check if start and end stations exist in the map
#     if start_station_name not in map or end_station_name not in map:
#         raise ValueError("Invalid start or end station")

    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Initialize the A* algorithm
    open_set = [(0, start_station)]                    # Priority queue for the open set
    came_from = {}                                     # Mapping of visited stations and their previous stations
    g_scores = {station_name: float('inf') for station_name in map}
    g_scores[start_station_name] = 0

    f_scores = {station_name: float('inf') for station_name in map}
    f_scores[start_station_name] = heuristic(start_station, end_station)
    total_distance = 0  # 总路程

    while open_set:
        _, current_station = heappop(open_set)

        if current_station == end_station:
            path = reconstruct_path(came_from, end_station)
            return reconstruct_path(came_from, end_station)

        for neighbor_station in current_station.links:
            temp_g_score = g_scores[current_station.name] + 1      # Distance between adjacent stations is 1

            if temp_g_score < g_scores[neighbor_station.name]:
                came_from[neighbor_station] = current_station
                g_scores[neighbor_station.name] = temp_g_score
                f_scores[neighbor_station.name] = temp_g_score + heuristic(neighbor_station, end_station)
                heappush(open_set, (f_scores[neighbor_station.name], neighbor_station))
        # 计算从起点到当前节点的累计路程
        total_distance = g_scores[current_station.name]
    raise ValueError("No path found")

#A*变体距离
def get_path2(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs A* on the map to find the shortest path between start_station_name and end_station_name.
    Args:
        start_station_name (str): The name of the starting station
        end_station_name (str): The name of the ending station
        map (Dict[str, Station]): Mapping between station names and station objects of the name,
                                  Please refer to the relevant comments in the build_data.py
                                  for the description of the Station class
    Returns:
        List[str]: A list of station names representing the shortest path
    """
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Initialize the A* algorithm
    open_set = [(0, start_station)]                    # Priority queue for the open set
    came_from = {}                                     # Mapping of visited stations and their previous stations
    g_scores = {station_name: float('inf') for station_name in map}
    g_scores[start_station_name] = 0

    f_scores = {station_name: float('inf') for station_name in map}
    f_scores[start_station_name] = heuristic1(start_station, end_station)  # 使用 heuristic_variant 替换 heuristic
    total_distance = 0  # 总路程

    while open_set:
        _, current_station = heappop(open_set)
        if current_station == end_station:
            path = reconstruct_path(came_from, end_station)
            return reconstruct_path(came_from, end_station)

        for neighbor_station in current_station.links:
            temp_g_score = g_scores[current_station.name] + 1      # Distance between adjacent stations is 1

            if temp_g_score < g_scores[neighbor_station.name]:
                came_from[neighbor_station] = current_station
                g_scores[neighbor_station.name] = temp_g_score
                f_scores[neighbor_station.name] = temp_g_score + heuristic1(neighbor_station, end_station)  # 使用 heuristic_variant 替换 heuristic
                heappush(open_set, (f_scores[neighbor_station.name], neighbor_station))
        # 计算从起点到当前节点的累计路程
        total_distance = g_scores[current_station.name]
    raise ValueError("No path found")

# Dijkstra
def get_path3(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
#     """
#     Runs Dijkstra's algorithm on the map to find the shortest path between start_station_name and end_station_name.
#     Args:
#         start_station_name (str): The name of the starting station
#         end_station_name (str): The name of the ending station
#         map (Dict[str, Station]): Mapping between station names and station objects
#     Returns:
#         List[str]: A list of station names representing the shortest path
#     """
    # Check if start and end stations exist in the map
    if start_station_name not in map or end_station_name not in map:
        raise ValueError("Invalid start or end station")

    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Initialize distances dictionary
    distances = {station_name: float('inf') for station_name in map}
    distances[start_station_name] = 0

    # Initialize priority queue for the open set
    open_set = [(0, start_station)]
    # Initialize mapping of visited stations and their previous stations
    came_from = {}

    while open_set:
        current_distance, current_station = heappop(open_set)

        if current_station == end_station:
            path = reconstruct_path(came_from, end_station)
            return path

        for neighbor_station in current_station.links:
            distance = current_distance + heuristic(current_station, neighbor_station)

            if distance < distances[neighbor_station.name]:
                distances[neighbor_station.name] = distance
                came_from[neighbor_station] = current_station
                heappush(open_set, (distance, neighbor_station))

    raise ValueError("No path found")

def reconstruct_path(came_from: Dict[Station, Station], current_station: Station) -> List[str]:
    """
    Reconstructs the path from the start station to the current station using the came_from dictionary.
    Args:
        came_from (Dict[Station, Station]): Mapping of visited stations and their previous stations
        current_station (Station): The current station
    Returns:
        List[str]: A list of station names representing the path from start to current station
    """
    path = [current_station.name]
    while current_station in came_from:
        current_station = came_from[current_station]
        path.append(current_station.name)
    path.reverse()
    return path

#欧式距离
def heuristic(start_station: Station, end_station: Station) -> float:
    """
    Calculates the Euclidean distance heuristic between two stations.
    Args:
        start_station (Station): The starting station
        end_station (Station): The ending station
    Returns:
        float: The Euclidean distance between the two stations
    """
    start_lat, start_lon = start_station.position
    end_lat, end_lon = end_station.position
    return (((start_lat - end_lat) ** 2 + (start_lon - end_lon) ** 2) ** 0.5)*111

#启发函数变化
def heuristic1(start_station: Station, end_station: Station) -> float:
    """
    Calculates the Manhattan distance heuristic between two stations.
    Args:
        start_station (Station): The starting station
        end_station (Station): The ending station
    Returns:
        float: The Manhattan distance between the two stations
    """
    start_lat, start_lon = start_station.position
    end_lat, end_lon = end_station.position
    # result=(abs(start_lat - end_lat) + abs(start_lon - end_lon))#曼哈顿距离
    result = (max(abs(start_lat - end_lat), abs(start_lon - end_lon)))*111#切比雪夫距离
    return result

#经纬度，算真实总距离
def heuristic2(start_station: Station, end_station: Station) -> float:
    """
    Calculates the Haversine distance heuristic between two stations.
    Args:
        start_station (Station): The starting station
        end_station (Station): The ending station
    Returns:
        float: The Haversine distance between the two stations
    """
    start_lat, start_lon = start_station.position
    end_lat, end_lon = end_station.position

    # Convert coordinates from degrees to radians
    lat1 = radians(start_lat)
    lon1 = radians(start_lon)
    lat2 = radians(end_lat)
    lon2 = radians(end_lon)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Radius of the Earth in kilometers

    return distance

#根据不同的要求改变getpath和heuristic函数
if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()
    # Add command-line arguments
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    #start calculate time
    start_time = time.time()
    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path2(start_station_name, end_station_name, stations)
    print(path)
    # calculate distance
    # distances = calculate_distances(stations)
    # print(distances)
    total_distance = 0
    for i in range(len(path) - 1):
        start_station = stations[path[i]]
        end_station = stations[path[i+1]]
        print(start_station.name, round(heuristic2(start_station, end_station), 4),"km",end_station.name)
        total_distance += heuristic2(start_station, end_station)

    print("总路程:", total_distance,"km")
    #time end
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码运行时间:", execution_time, "秒")
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)


