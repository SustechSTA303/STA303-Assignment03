from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue
import math
from math import radians, cos, sin, asin, sqrt
import time
from collections import deque  

def euclidean_distance(start_station_position, end_station_position):
    
    lat1 = start_station_position[0]
    lon1 = start_station_position[1]
    lat2 = end_station_position[0]
    lon2 = end_station_position[1]
    
    # 将十进制度数转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算平面坐标系中的坐标
    x1 = math.cos(lat1) * math.cos(lon1)
    y1 = math.cos(lat1) * math.sin(lon1)
    x2 = math.cos(lat2) * math.cos(lon2)
    y2 = math.cos(lat2) * math.sin(lon2)

    # 计算欧几里得距离
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance
    
# Implement the following function
def Astar(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
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
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the  following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    g_values = {station: float('inf') for station in map.keys()}
    g_values[start_station_name] = 0
    
    parents = {}
    
    open_list = PriorityQueue()
    open_list.put((g_values[start_station_name] + euclidean_distance(start_station.position, end_station.position), start_station_name))
    
    closed_list = set()
    
    while not open_list.empty():
        current_node = open_list.get()[1]
        
        if current_node == end_station_name:
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            return path[::-1]
        
        closed_list.add(current_node)
        
        for neighbor in map[current_node].links:
            neighbor_name = neighbor.name
            distance = euclidean_distance(neighbor.position, map[current_node].position)
            
            temp_g = g_values[current_node] + distance
            
            if neighbor_name not in closed_list or temp_g < g_values[neighbor_name]:
                parents[neighbor_name] = current_node
                g_values[neighbor_name] = temp_g
                
                priority = temp_g + euclidean_distance(map[neighbor_name].position, end_station.position)
                open_list.put((priority, neighbor_name))
    
    return []

def Dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    Runs Dijkstra's algorithm on the map, finds the shortest path between start and end stations.
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

    g_values = {station: float('inf') for station in map.keys()}
    g_values[start_station_name] = 0

    parents = {}

    open_list = PriorityQueue()
    open_list.put((g_values[start_station_name], start_station_name))

    closed_list = set()

    while not open_list.empty():
        current_node = open_list.get()[1]

        if current_node == end_station_name:
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            return path[::-1]

        closed_list.add(current_node)

        for neighbor in map[current_node].links:
            neighbor_name = neighbor.name
            distance = euclidean_distance(neighbor.position, map[current_node].position)

            temp_g = g_values[current_node] + distance

            if neighbor_name not in closed_list or temp_g < g_values[neighbor_name]:
                parents[neighbor_name] = current_node
                g_values[neighbor_name] = temp_g

                open_list.put((g_values[neighbor_name], neighbor_name))

    return []


def SPFA(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    Runs SPFA algorithm on the map, finds the shortest path between start and end stations.
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

    
    distance = {station: float('inf') for station in map.keys()}
    distance[start_station_name] = 0
    
    closed_list = set()

    parents = {}

    queue = deque()
    queue.append(start_station_name)

    while queue:
        current_node = queue.popleft()

        if current_node == end_station_name:
            path = []
            node = current_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            return path[::-1]

        for neighbor in map[current_node].links:
            neighbor_name = neighbor.name
            edge_weight = euclidean_distance(neighbor.position, map[current_node].position)
            
            if distance[current_node] + edge_weight < distance[neighbor_name]:
                distance[neighbor_name] = distance[current_node] + edge_weight
                parents[neighbor_name] = current_node

                if neighbor_name not in queue:
                    queue.append(neighbor_name)

    return []

def BF(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

    g_values = {station: float('inf') for station in map.keys()}
    g_values[start_station_name] = 0

    parents = {}

    open_list = PriorityQueue()
    open_list.put((g_values[start_station_name], start_station_name))

    closed_list = set()

    while not open_list.empty():
        current_station_name = open_list.get()[1]

        if current_station_name == end_station_name:
            break

        closed_list.add(current_station_name)

        for neighbor in map[current_station_name].links:
            neighbor_name = neighbor.name
            distance = euclidean_distance(neighbor.position, map[current_station_name].position)
            
            if neighbor_name in closed_list:
                continue

            new_g = g_values[current_station_name] + distance
            if new_g < g_values[neighbor_name]:
                g_values[neighbor_name] = new_g
                parents[neighbor_name] = current_station_name
                open_list.put((g_values[neighbor_name], neighbor_name))


    path = []
    current = end_station_name
    while current != start_station_name:
        path.append(current)
        current = parents[current]
    path.append(start_station_name)
    path.reverse()

    return path


if __name__ == '__main__':
    start_time = time.time()
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
    path = Astar(start_station_name, end_station_name, stations)
    plot_path(path, './visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    end_time = time.time()  # 记录程序结束时间
    print(f"程序运行时间: {end_time - start_time:.6f}秒")  # 输出程序运行时间
