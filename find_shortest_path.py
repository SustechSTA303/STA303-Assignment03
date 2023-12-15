import argparse
import time

from build_data import Station, build_data
from plot_underground_path import plot_multiple_paths


# Manhattan Distance
def manhattan_distance(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    return abs(lat1 - lat2) + abs(lon1 - lon2)


def chebyshev_distance(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    return max(abs(lat1 - lat2), abs(lon1 - lon2))


def euclidean_distance(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def octile_distance(station1, station2):
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    dx = abs(lat1 - lat2)
    dy = abs(lon1 - lon2)
    return max(dx, dy) + 0.414 * min(dx, dy)


# Implement the following function
from typing import List, Callable


# Modify the get_path function to accept a heuristic function as a parameter
def get_path_A(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic: Callable) -> List[
    str]:
    time_start = time.perf_counter()
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = [start_station]
    came_from = {}
    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    f_score = {station: float('inf') for station in map.values()}
    f_score[start_station] = heuristic(start_station, end_station)

    while open_set:
        current_station = min(open_set, key=lambda station: f_score[station])

        if current_station == end_station:
            path = []
            while current_station in came_from:
                path.append(current_station.name)
                current_station = came_from[current_station]
            path.append(start_station_name)
            print("Total time of " + heuristic.__name__ + ": " + str(time.perf_counter() - time_start) + "s")
            return path[::-1]

        open_set.remove(current_station)
        for neighbor_station in current_station.links:
            tentative_g_score = g_score[current_station] + heuristic(neighbor_station, current_station)

            if tentative_g_score < g_score[neighbor_station]:
                came_from[neighbor_station] = current_station
                g_score[neighbor_station] = tentative_g_score
                f_score[neighbor_station] = g_score[neighbor_station] + heuristic(neighbor_station, end_station)
                if neighbor_station not in open_set:
                    open_set.append(neighbor_station)
    return []


def get_path_dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic: Callable) -> List[str]:
    time_start = time.perf_counter()
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = [start_station]
    came_from = {}
    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    f_score = {station: float('inf') for station in map.values()}
    f_score[start_station] = heuristic(start_station, end_station)

    while open_set:
        current_station = min(open_set, key=lambda station: f_score[station])

        if current_station == end_station:
            path = []
            while current_station in came_from:
                path.append(current_station.name)
                current_station = came_from[current_station]
            path.append(start_station_name)
            print("Total time of " + "Dijkstra" + ": " + str(time.perf_counter() - time_start) + "s")
            return path[::-1]

        open_set.remove(current_station)
        for neighbor_station in current_station.links:
            tentative_g_score = g_score[current_station] + heuristic(neighbor_station, current_station)

            if tentative_g_score < g_score[neighbor_station]:
                came_from[neighbor_station] = current_station
                g_score[neighbor_station] = tentative_g_score
                f_score[neighbor_station] = g_score[neighbor_station]
                if neighbor_station not in open_set:
                    open_set.append(neighbor_station)
    return []


def get_path_bestFirst(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic: Callable) -> List[str]:
    time_start = time.perf_counter()
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = [start_station]
    came_from = {}
    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    f_score = {station: float('inf') for station in map.values()}
    f_score[start_station] = heuristic(start_station, end_station)

    while open_set:
        current_station = min(open_set, key=lambda station: f_score[station])

        if current_station == end_station:
            path = []
            while current_station in came_from:
                path.append(current_station.name)
                current_station = came_from[current_station]
            path.append(start_station_name)
            print("Total time of " + "Best First" + ": " + str(time.perf_counter() - time_start) + "s")
            return path[::-1]

        open_set.remove(current_station)
        for neighbor_station in current_station.links:
            tentative_g_score = heuristic(neighbor_station, end_station)

            if tentative_g_score < g_score[neighbor_station]:
                came_from[neighbor_station] = current_station
                g_score[neighbor_station] = tentative_g_score
                f_score[neighbor_station] = heuristic(neighbor_station, end_station)
                if neighbor_station not in open_set:
                    open_set.append(neighbor_station)
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

    heuristic_names = ["Path Manhattan", "Path Chebyshev", "Path Euclidean", "Path Octile", "Path Dijkstra", "Path Best First"]

    path_manhattan = get_path_A(start_station_name, end_station_name, stations, heuristic=manhattan_distance)

    path_chebyshev = get_path_A(start_station_name, end_station_name, stations, heuristic=chebyshev_distance)

    path_euclidean = get_path_A(start_station_name, end_station_name, stations, heuristic=euclidean_distance)

    path_octile = get_path_A(start_station_name, end_station_name, stations, heuristic=octile_distance)

    path_dijkstra = get_path_dijkstra(start_station_name, end_station_name, stations, heuristic=euclidean_distance)

    path_bestFirst = get_path_bestFirst(start_station_name, end_station_name, stations, heuristic=euclidean_distance)

    # Assuming paths are collected into a list
    all_paths = [path_manhattan, path_chebyshev, path_euclidean, path_octile, path_dijkstra, path_bestFirst]

    # Visualize multiple paths on the same map
    plot_multiple_paths(all_paths, 'visualization_underground/multiple_paths_comparison.html', stations, heuristic_names, underground_lines)
