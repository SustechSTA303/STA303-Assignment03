from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
from math import sqrt
import numpy as np
from queue import PriorityQueue
from math import sqrt
import time
import random
import pandas as pd

def random_station_name(stations):
    return random.choice(list(stations.keys()))

# 为a*算法设计的四种启发函数

def euclidean_heuristic(station_a, station_b):
    (lat_a, lon_a), (lat_b, lon_b) = station_a.position, station_b.position
    return sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2)

def manhattan_heuristic(station_a, station_b):
    (lat_a, lon_a), (lat_b, lon_b) = station_a.position, station_b.position
    return abs(lat_a - lat_b) + abs(lon_a - lon_b)

def chebyshev_heuristic(station_a, station_b):
    (lat_a, lon_a), (lat_b, lon_b) = station_a.position, station_b.position
    return max(abs(lat_a - lat_b), abs(lon_a - lon_b))

def cosine_similarity_heuristic(station_a, station_b):
    # Now 'np' is defined, because you have imported numpy at the top of the file
    (lat_a, lon_a), (lat_b, lon_b) = station_a.position, station_b.position
    vector_a = np.array([lat_a, lon_a])
    vector_b = np.array([lat_b, lon_b])
    cosine_similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    # Convert cosine similarity to a distance measure
    return 1 - cosine_similarity

# 设计加入dijkstra算法，一起比较

def dijkstra_heuristic(station_a, station_b):
    # Dijkstra算法不使用启发式，因此我们可以返回0
    return 0

def dijkstra_path(start_station_name, end_station_name, stations):
    return get_path(start_station_name, end_station_name, stations, dijkstra_heuristic)

# a*算法的实现

def get_path(start_station_name, end_station_name, stations, heuristic_func):
    start_station = stations[start_station_name]
    end_station = stations[end_station_name]
    open_set = PriorityQueue()
    open_set.put((0, start_station))

    came_from = {}
    g_score = {station: float('inf') for station in stations.values()}
    g_score[start_station] = 0

    f_score = {station: float('inf') for station in stations.values()}
    f_score[start_station] = heuristic_func(start_station, end_station)

    open_set_hash = {start_station}

    while not open_set.empty():
        current = open_set.get()[1]
        open_set_hash.remove(current)

        if current == end_station:
            path = []
            while current in came_from:
                path.append(current.name)
                current = came_from[current]
            path.append(start_station.name)
            return path[::-1]

        for neighbor in current.links:
            tentative_g_score = g_score[current] + heuristic_func(current, neighbor)  # 使用传递的启发函数

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_func(neighbor, end_station)  # 使用传递的启发函数
                if neighbor not in open_set_hash:
                    open_set.put((f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return []

def distance_between_stations(station_a, station_b):
    (lat_a, lon_a), (lat_b, lon_b) = station_a.position, station_b.position
    # 使用欧几里得距离计算两个站点间的距离
    return sqrt((lat_a - lat_b) ** 2 + (lon_a - lon_b) ** 2)

def calculate_path_length(path, stations):
    # 计算路径的实际长度
    length = 0
    for i in range(len(path) - 1):
        station_a = stations[path[i]]
        station_b = stations[path[i + 1]]
        length += distance_between_stations(station_a, station_b)
    return length

if __name__ == '__main__':
    # 导入必要的模块
    import argparse
    
#     # 创建ArgumentParser对象
#     parser = argparse.ArgumentParser()
#     # 添加命令行参数
#     parser.add_argument('start_station_name', type=str, help='起始站点名称')
#     parser.add_argument('end_station_name', type=str, help='结束站点名称')
#     args = parser.parse_args()
#     start_station_name = args.start_station_name
#     end_station_name = args.end_station_name

    # 导入其他必要的模块
    from build_data import build_data
    from plot_underground_path import plot_path

    # 获取站点和地铁线路数据
    stations, underground_lines = build_data()
    
    heuristics = {
        'euclidean': euclidean_heuristic,
        'manhattan': manhattan_heuristic,
        'chebyshev': chebyshev_heuristic,
        'cosine': cosine_similarity_heuristic,
        'dijkstra': dijkstra_heuristic
    }
    
    # 运行算法的次数
    num_runs = 10
    heuristic_performance = {name: [] for name in heuristics}
    
    for i in range(num_runs):
        start_station_name = random_station_name(stations)
        end_station_name = random_station_name(stations)
        while start_station_name == end_station_name:
            end_station_name = random_station_name(stations)

        print(f"|---起始站: {start_station_name}, 终点站: {end_station_name}, 第{i+1}次实验---|")

        results = {}
        # Prepare data for the table
        table_data = []

        for name, heuristic in heuristics.items():
            start_time = time.time()
            path = get_path(start_station_name, end_station_name, stations, heuristic)
            end_time = time.time()
            duration = end_time - start_time
            path_length = calculate_path_length(path, stations)
            results[name] = (duration, path_length, path)
            table_data.append({'Heuristic': name, 'Time (seconds)': f"{duration:.5f}", 'Path Length': f"{path_length:.4f}"})

        # Display results in a table
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))

        # Choose the best heuristic based on time and then path length
        best_heuristic = sorted(results.items(), key=lambda x: (x[1][0], x[1][1]))[0][0]
        print(f"这次迭代最优的启发函数是: {best_heuristic}, 路径: {results[best_heuristic][2]}")
        heuristic_performance[best_heuristic].append(results[best_heuristic][0])
        plot_path(results[best_heuristic][2], f'visualization_underground/best_path_iteration_{i+1}.html', stations, underground_lines)
        
    # 确定整体最优的启发函数
    overall_best = max(heuristic_performance, key=lambda x: sum(heuristic_performance[x]))
    print(f"总体最优的启发函数是: {overall_best}")

