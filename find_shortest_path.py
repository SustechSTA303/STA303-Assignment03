from typing import List
import argparse
import csv
import os
import time
from math import radians, cos, sin, asin, sqrt
import numpy as np
import math
import heapq
from collections import deque
from build_data import build_data
from build_data import Station
from plot_underground_path import plot_path

#更详细的实现和数据请参照 AI-A3_Jupyterlab.ipynb
def euclidean_distance(station_a, station_b):
    ax, ay = station_a.position
    bx, by = station_b.position
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def manhattan_distance(station_a, station_b):
    ax, ay = station_a.position
    bx, by = station_b.position
    return abs(ax - bx) + abs(ay - by)

def simple_distance(station_a, station_b):
    ax, ay = station_a.position
    bx, by = station_b.position
    return max(abs(ax - bx), abs(ay - by))

def cosine_similarity_distance(station_a, station_b):
    (lat_a, lon_a), (lat_b, lon_b) = station_a.position, station_b.position
    vector_a = np.array([lat_a, lon_a])
    vector_b = np.array([lat_b, lon_b])
    cosine_similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    # Convert cosine similarity to a distance measure
    return 1 - cosine_similarity

def haversine_distance(station_a, station_b):
    lat1, lon1 = station_a.position
    lat2, lon2 = station_b.position
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r

def astar_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_func) -> List[str]:
    open_set = set([start_station_name])
    g_score = {station: float('infinity') for station in map}
    g_score[start_station_name] = 0

    # 初始化 f_score 字典
    f_score = {station: float('infinity') for station in map}
    f_score[start_station_name] = heuristic_func(map[start_station_name], map[end_station_name])

    path = {start_station_name: None}

    while open_set:
        current_station = min(open_set, key=lambda station: f_score[station])
        if current_station == end_station_name:
            break

        open_set.remove(current_station)
        for neighbor in map[current_station].neighbors():
            tentative_g_score = g_score[current_station] + map[current_station].get_distance(neighbor)
            if tentative_g_score < g_score[neighbor.name]:
                path[neighbor.name] = current_station
                g_score[neighbor.name] = tentative_g_score
                f_score[neighbor.name] = tentative_g_score + heuristic_func(map[neighbor.name], map[end_station_name])
                if neighbor.name not in open_set:
                    open_set.add(neighbor.name)
    current = end_station_name
    shortest_path = []
    while current is not None:
        shortest_path.append(current)
        current = path[current]
    return shortest_path[::-1]  # 反转路径

def dijkstra_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    distances = {station: float('infinity') for station in map}
    distances[start_station_name] = 0
    priority_queue = [(0, start_station_name)]
    path = {start_station_name: None}

    while priority_queue:
        current_distance, current_station = heapq.heappop(priority_queue)
        if current_station == end_station_name:
            break

        for neighbor in map[current_station].neighbors():
            distance = current_distance + map[current_station].get_distance(neighbor)
            if distance < distances[neighbor.name]:
                distances[neighbor.name] = distance
                heapq.heappush(priority_queue, (distance, neighbor.name))
                path[neighbor.name] = current_station

    # Backtrack from end_station_name to start_station_name
    current = end_station_name
    shortest_path = []
    while current is not None:
        shortest_path.append(current)
        current = path[current]
    return shortest_path[::-1]  # Reverse the path

def dfs_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    stack = [(start_station_name, [start_station_name])]
    visited = set()

    while stack:
        current_station, path = stack.pop()

        if current_station == end_station_name:
            return path

        if current_station not in visited:
            visited.add(current_station)
            neighbors = map[current_station].neighbors()

            # Sort neighbors based on a heuristic function
            sorted_neighbors = sorted(neighbors, key=lambda neighbor: euclidean_distance(map[neighbor.name], map[end_station_name]))

            for neighbor in sorted_neighbors:
                stack.append((neighbor.name, path + [neighbor.name]))

    return []

def greedy_best_first_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    open_set = set([start_station_name])
    path = [start_station_name]

    while open_set:
        current_station = min(open_set, key=lambda station: euclidean_distance(map[station], map[end_station_name]))
        if current_station == end_station_name:
            return path

        open_set.remove(current_station)
        neighbors = map[current_station].neighbors()
        nearest_neighbor = min(neighbors, key=lambda neighbor: euclidean_distance(map[neighbor.name], map[end_station_name]))
        path.append(nearest_neighbor.name)
        open_set.add(nearest_neighbor.name)

    return []
#此算法不适合规定了路线的

def bfs_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    queue = deque([(start_station_name, [start_station_name])])  # 使用队列存储待探索的节点和路径
    visited = set()

    while queue:
        current_station, path = queue.popleft()

        if current_station == end_station_name:
            return path

        if current_station not in visited:
            visited.add(current_station)

            neighbors = map[current_station].neighbors()
            for neighbor in neighbors:
                if neighbor.name not in visited:
                    queue.append((neighbor.name, path + [neighbor.name]))

    return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    stations, underground_lines = build_data()
    path = astar_get_path(start_station_name, end_station_name, stations, euclidean_distance)
    print(path)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
