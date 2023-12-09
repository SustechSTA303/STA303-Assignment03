import math
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time
from queue import PriorityQueue
from collections import deque
import heapq


# 哈弗辛公式（欧几里得距离）
# def heuristic_Euclidean(station1, station2):
#     R = 6371  # 地球半径，单位为千米
#     lat1, lon1 = map(math.radians, station1.position)
#     lat2, lon2 = map(math.radians, station2.position)
#
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#
#     a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#
#     return R * c
#

def heuristic_Euclidean(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position

    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# 曼哈顿距离（适应地球曲率）
# def heuristic_Manhattan(station1, station2):
#     R = 6371  # 地球半径，单位为千米
#
#     lat1, lon1 = map(math.radians, station1.position)
#     lat2, lon2 = map(math.radians, station2.position)
#
#     # 纬度变化距离
#     dlat = lat2 - lat1
#     lat_distance = R * 2 * math.atan2(math.sqrt(math.sin(dlat / 2) ** 2), math.sqrt(1 - math.sin(dlat / 2) ** 2))
#
#     # 经度变化距离（在相同纬度）
#     avg_lat = (lat1 + lat2) / 2
#     dlon = lon2 - lon1
#     lon_distance = R * math.cos(avg_lat) * 2 * math.atan2(math.sqrt(math.sin(dlon / 2) ** 2),
#                                                           math.sqrt(1 - math.sin(dlon / 2) ** 2))
#
#     return lat_distance + lon_distance


def heuristic_Manhattan(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position

    return abs(lat1 - lat2) + abs(lon1 - lon2)


def heuristic_Constant(station1, station2):
    return 0


# Implement the following function
def astar_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_func) -> List[
    str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = PriorityQueue()
    open_set.put((0, start_station_name))
    came_from = {start_station_name: None}
    g_score = {station: float('inf') for station in map}
    g_score[start_station_name] = 0

    while not open_set.empty():
        _, current_name = open_set.get()
        current_station = map[current_name]

        if current_name == end_station_name:
            path = []
            while current_name:
                path.append(current_name)
                current_name = came_from[current_name]
            return path[::-1]

        for neighbor in current_station.links:
            neighbor_name = neighbor.name
            tentative_g_score = g_score[current_name] + heuristic_func(neighbor,current_station)

            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score = tentative_g_score + heuristic_func(neighbor, end_station)
                open_set.put((f_score, neighbor_name))

    return []


def bfs_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_func) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    queue = deque([start_station])
    came_from = {start_station: None}

    while queue:
        current = queue.popleft()
        if current == end_station:
            break

        for neighbor in current.links:
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current

    if end_station not in came_from:
        return []  # Path not found

    # Reconstruct path
    path = []
    while current:
        path.append(current.name)
        current = came_from[current]
    return path[::-1]


def dfs_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_func) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    stack = [start_station]
    came_from = {start_station: None}
    visited = set()  # 用于跟踪已访问的站点

    while stack:
        current = stack.pop()
        visited.add(current)  # 将当前站点标记为已访问

        if current == end_station:
            break

        for neighbor in current.links:
            if neighbor not in came_from and neighbor not in visited:
                stack.append(neighbor)
                came_from[neighbor] = current

    if end_station not in came_from:
        return []  # Path not found

    # Reconstruct path
    path = []
    while current:
        path.append(current.name)
        current = came_from[current]
    return path[::-1]



def dijkstra_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_func) -> List[
    str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    pq = [(0, start_station_name)]  # 使用站点名称而非对象
    came_from = {start_station_name: None}
    cost_so_far = {start_station_name: 0}

    while pq:
        current_cost, current_name = heapq.heappop(pq)
        current = map[current_name]

        if current == end_station:
            break

        for neighbor in current.links:
            neighbor_name = neighbor.name
            new_cost = current_cost + heuristic_func(current, neighbor)
            if neighbor_name not in cost_so_far or new_cost < cost_so_far[neighbor_name]:
                cost_so_far[neighbor_name] = new_cost
                priority = new_cost
                heapq.heappush(pq, (priority, neighbor_name))
                came_from[neighbor_name] = current_name

    if end_station_name not in came_from:
        return []

    # Reconstruct path using station names
    path = []
    current_name = end_station_name
    while current_name:
        path.append(current_name)
        current_name = came_from[current_name]
    return path[::-1]


def evaluate_algorithm(algorithm, heuristic, stations, start_station_name=None):
    total_duration = 0
    total_path_length = 0

    start_stations = stations.keys() if start_station_name is None else [start_station_name]

    for start_station in start_stations:
        for end_station_name in stations:
            if start_station != end_station_name:
                start_time = time.time()
                path = algorithm(start_station, end_station_name, stations, heuristic)
                end_time = time.time()

                total_duration += end_time - start_time
                path_length = calculate_path_length(path, stations)
                total_path_length += path_length

    return total_duration, total_path_length


def calculate_path_length(path, stations):
    path_length = 0
    for i in range(len(path) - 1):
        station1 = stations[path[i]]
        station2 = stations[path[i + 1]]
        path_length += heuristic_Euclidean(station1, station2)
    return path_length


def find_transfer_stations(stations, underground_lines):
    transfer_stations = []

    for station_name, station in stations.items():
        lines = set()

        for neighbor in station.links:
            for line_number, line_info in underground_lines.items():
                if station.name in line_info['stations'] and neighbor.name in line_info['stations']:
                    lines.add(line_number)

        if len(lines) > 1:
            transfer_stations.append(station_name)

    return transfer_stations


def print_path_details(start_station_name, end_station_name, stations):
    # A* with Euclidean Heuristic
    path = astar_get_path(start_station_name, end_station_name, stations, heuristic_Euclidean)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f"A* with Euclidean Heuristic Path: {path}")

    # A* with Manhattan Heuristic
    path = astar_get_path(start_station_name, end_station_name, stations, heuristic_Manhattan)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f"A* with Manhattan Heuristic Path: {path}")

    # BFS
    path = bfs_get_path(start_station_name, end_station_name, stations, heuristic_Constant)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f"BFS Path: {path}")

    # DFS
    path = dfs_get_path(start_station_name, end_station_name, stations, heuristic_Constant)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f"DFS Path: {path}")

    # Dijkstra
    path = dijkstra_get_path(start_station_name, end_station_name, stations, heuristic_Constant)
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f"Dijkstra Path: {path}")


if __name__ == '__main__':
    stations, underground_lines = build_data()
    # start_station_name = "Holborn"
    # start_station_name = None
    # transfer_stations = find_transfer_stations(stations, underground_lines)
    # print(len(transfer_stations))
    # print("Transfer Stations:", transfer_stations)

    print_path_details("Holborn", "Rayners Lane", stations)


    for i in range(5):
        a_star_duration, a_star_path_length = evaluate_algorithm(astar_get_path, heuristic_Euclidean,
                                                                 stations, None)  # warm-up！

    # 调用函数以输出从 "Holborn" 到 "Finchley Road" 的路径


    # 评估 A* 算法
    a_star_duration, a_star_path_length = evaluate_algorithm(astar_get_path, heuristic_Euclidean,
                                                             stations, None)
    print(
        f"A* Algorithm with heuristic function as Euclidean - Total Duration: {a_star_duration}s, Total Path Length: {a_star_path_length}km")

    a_star_duration, a_star_path_length = evaluate_algorithm(astar_get_path, heuristic_Manhattan,
                                                             stations, None)
    print(
        f"A* Algorithm with heuristic function as Manhattan - Total Duration: {a_star_duration}s, Total Path Length: {a_star_path_length}km")

    # 评估 BFS
    bfs_duration, bfs_path_length = evaluate_algorithm(bfs_get_path, heuristic_Constant, stations, None)
    print(f"BFS Algorithm - Total Duration: {bfs_duration}s, Total Path Length: {bfs_path_length}km")

    # 评估 DFS
    dfs_duration, dfs_path_length = evaluate_algorithm(dfs_get_path, heuristic_Constant, stations, None)
    print(f"DFS Algorithm - Total Duration: {dfs_duration}s, Total Path Length: {dfs_path_length}km")

    # 评估 Dijkstra
    dijkstra_duration, dijkstra_path_length = evaluate_algorithm(dijkstra_get_path, heuristic_Constant, stations,
                                                                 None)
    print(f"Dijkstra Algorithm - Total Duration: {dijkstra_duration}s, Total Path Length: {dijkstra_path_length}km")

    # 创建ArgumentParser对象
    # parser = argparse.ArgumentParser()
    # # 添加命令行参数
    # parser.add_argument('start_station_name', type=str, help='start_station_name')
    # parser.add_argument('end_station_name', type=str, help='end_station_name')
    # args = parser.parse_args()
    # start_station_name = args.start_station_name
    # end_station_name = args.end_station_name

    # end_station_name = "Hounslow East"
    # The relevant descriptions of stations and underground_lines can be found in the build_data.py

    # start_time = time.time()

    # path = astar_get_path(start_station_name, end_station_name, stations, heuristic_Euclidean)
    # end_time = time.time()

    # duration = end_time - start_time;
    # print(f"time to find the path is: {duration}")
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    # plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
