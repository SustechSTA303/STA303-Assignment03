from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
from queue import PriorityQueue
import random

from typing import List

def estimate_heuristic(station, goal_station):
    # 使用直线距离作为启发式（估计）代价
    return distance(station, goal_station)

def distance(station1, station2):
    # 计算两个站点之间的直线距离（欧几里得距离）
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    distance_km = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return distance_km

from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
from queue import PriorityQueue

def weighted_a_star(start_station_name: str, end_station_name: str, map: dict[str, Station], weight: float = 3.0) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = PriorityQueue()
    open_set.put((0, start_station, [start_station]))

    closed_set = set()

    while not open_set.empty():
        current_cost, current_station, current_path = open_set.get()

        if current_station == end_station:
            path_names = [station.name for station in current_path]
            return path_names  # 找到路径

        closed_set.add(current_station)

        for neighbor in current_station.links:
            if neighbor not in closed_set:
                new_cost = current_cost + distance(current_station, neighbor)
                heuristic = weight * estimate_heuristic(neighbor, end_station)
                total_cost = new_cost + heuristic
                new_path = current_path + [neighbor]

                open_set.put((total_cost, neighbor, new_path))

    return None  # 没有找到路径

# 其余代码保持不变

if __name__ == '__main__':
    # ... (unchanged code for ArgumentParser, build_data, etc.)
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()
    
    # Use the new get_path_prim function
    prim_path = weighted_a_star(start_station_name, end_station_name,stations)
    print(f"Prim's Path: {prim_path}")

    # visualization the Prim's path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(prim_path, 'visualization_underground/my_prim_path_in_London_railway.html', stations, underground_lines)
