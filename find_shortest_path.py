from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os

## 定义哈弗辛距离函数来计算真实距离
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_of_earth = 6371.0
    distance = radius_of_earth * c
    return distance

# 定义曼哈顿距离函数来计算估计距离
def manhattan_distance_small(node_lat, node_lon, goal_lat, goal_lon):
    return  (abs(node_lat - goal_lat) + abs(node_lon - goal_lon))

def manhattan_distance_large(node_lat, node_lon, goal_lat, goal_lon):
    return 6666 * (abs(node_lat - goal_lat) + abs(node_lon - goal_lon))

def manhattan_distance(node_lat, node_lon, goal_lat, goal_lon):
    return 111 * (abs(node_lat - goal_lat) + abs(node_lon - goal_lon))

# 定义对角线距离函数来计算估计距离
def diagonal_distance(node_lat, node_lon, goal_lat, goal_lon):
    return 156 * max(abs(node_lat - goal_lat), abs(node_lon - goal_lon))

# 定义欧几里得距离函数来计算估计距离
def euclidean_distance(node_lat, node_lon, goal_lat, goal_lon):
    return 156 * math.sqrt((node_lat - goal_lat)**2 + (node_lon - goal_lon)**2)


def get_path(start_station_name, end_station_name, map, algorithm, heuristic):
    start_time = time.time()
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    if start_station.name not in map:
        raise TypeError(start_station.name + ' not found in graph!')
    if end_station.name not in map:
        raise TypeError(end_station.name + ' not found in graph!')

    distances = {station: float('infinity') for station in stations.keys()}
    distances[start_station.name] = 0

    queue = PriorityQueue()
    queue.put((0, [start_station.name]))
    while not queue.empty():
        priority, path = queue.get()
        if path[-1] == end_station.name:
            end_time = time.time()
            execution_time = end_time - start_time
            return distances[end_station.name], path, execution_time
        for neighbor in map[path[-1]].links:
            lon1 = map[path[-1]].position[0]
            lat1 = map[path[-1]].position[1]
            lon2 = map[neighbor.name].position[0]
            lat2 = map[neighbor.name].position[1]
            lon_end = map[end_station.name].position[0]
            lat_end = map[end_station.name].position[1]
            next_distance = distances[path[-1]] + haversine_distance(lat1, lon1, lat2, lon2)
            if next_distance < distances[neighbor.name]:
                distances[neighbor.name] = next_distance
                if (algorithm == "Dijkstra’s Algorithm"):
                    priority = next_distance
                elif (algorithm == "Greedy Best First Search"):
                    priority = heuristic(lat2, lon2, lat_end, lon_end)
                elif (algorithm == "A*"):
                    priority = next_distance + heuristic(lat2, lon2, lat_end, lon_end)
                queue.put((priority, path + [neighbor.name]))
    return None

def plot():
    folder_name = "image"
    file_name = f"{start_station_name}_to_{end_station_name}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_ylabel('Cost (km)', color=color)
    ax1.plot(all_algorithm, all_costs, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time (microseconds)', color=color)
    ax2.plot(all_algorithm, all_times, color=color, marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xticks(np.arange(len(all_algorithm)))
    ax1.set_xticklabels(all_algorithm, rotation=45, ha='right')
    fig.tight_layout()
    file_path = os.path.join(folder_name, file_name + ".png")
    plt.savefig(file_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name',default=None, nargs='?')
    parser.add_argument('end_station_name', type=str, help='end_station_name',default=None, nargs='?')


    args = parser.parse_args()

    # 定义算法和距离函数的数组
    algorithms = [
        ("Dijkstra’s Algorithm", None),
        ("Greedy Best First Search", haversine_distance),
        ("Greedy Best First Search", manhattan_distance),
        ("Greedy Best First Search", diagonal_distance),
        ("Greedy Best First Search", euclidean_distance),
        ("A*", haversine_distance),
        ("A*", manhattan_distance_small),
        ("A*", manhattan_distance_large),
        ("A*", manhattan_distance),
        ("A*", diagonal_distance),
        ("A*", euclidean_distance)
    ]
    results = {}
    all_algorithm = ["Dijkstra","Greedy(haversine)","Greedy(manhattan)","Greedy(diagonal)","Greedy(euclidean)","A*(haversine)","A*(manhattan_small)","A*(manhattan_large)","A*(manhattan)","A*(diagonal)","A*(euclidean)"]
    all_costs = []
    all_times = []
    stations, underground_lines = build_data()


    if args.start_station_name is not None and args.end_station_name is not None:
        start_station_name = args.start_station_name
        end_station_name = args.end_station_name
    else:
        print("None input, random select two stations")
        random_stations = random.sample(list(stations.keys()), 2)
        start_station_name = random_stations[0]
        end_station_name = random_stations[1]

    for algorithm, heuristic in algorithms:
        cost, path, execution_time = get_path(start_station_name, end_station_name, stations, algorithm, heuristic)
        results[algorithm, heuristic] = (cost, path, execution_time)

    print(start_station_name, "->", end_station_name, ": ")
    for result_key, result_value in results.items():
        algorithm, heuristic = result_key
        cost, path, execution_time = result_value
        all_costs.append(round(cost, 2))
        all_times.append(round(execution_time* 1000000, 2))

        file_name = f"path_{algorithm.lower()}_{heuristic.__name__.lower() if heuristic else 'none'}"
        plot_path(path, f'visualization_underground/{file_name}.html', stations, underground_lines)
        print(f"{algorithm} ({heuristic.__name__ if heuristic else 'No Heuristic'}):")
        print(f"  Cost: {cost:.2f}km")
        print(f"  Path: {path}")
        print(f"  Execution Time: {execution_time * 1000000:.2f} microseconds")
        print()
    max_algorithm_length = max(len(algorithm) for algorithm in all_algorithm)
    print(start_station_name, "--", end_station_name, ": ")
    for algorithm, cost, time in zip(all_algorithm, all_costs, all_times):
        print(f"{algorithm.ljust(max_algorithm_length)}: Distance: {cost:.2f}, Time: {time:.2f}")
    plot()



