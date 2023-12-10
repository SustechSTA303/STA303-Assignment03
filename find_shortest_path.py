import time
from typing import List, Tuple
from heapq import heappop, heappush

from plot_underground_path import plot_path
from build_data import Station, build_data
from heuristics_function import Euclidean
import argparse


#* A star algorithm
def A_star(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    runs a star on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): str The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
    Please refer to the relevant comments in the build_data.py for the description of the Station class
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    # print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    # print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

    #todo implement your code here
    open_set = [(0, start_station)]
    closed_set = set()
    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    came_from = {}

    start_time = time.time()
    while open_set:
        current_cost, current_station = heappop(open_set)

        if current_station == end_station:
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            end_time = time.time()
            return path, end_time-start_time

        closed_set.add(current_station)

        for neighbor in current_station.links:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_station] + Euclidean(current_station, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_station
                g_score[neighbor] = tentative_g_score
                heappush(open_set, (tentative_g_score + Euclidean(neighbor, end_station), neighbor))
    return [],0.0

#* Dijkstra algorithm
def dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> Tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = [(0, start_station)]
    closed_set = set()
    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    came_from = {}
    start_time = time.time()

    while open_set:
        current_cost, current_station = heappop(open_set)

        if current_station == end_station:
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            end_time = time.time()
            return path, end_time - start_time

        closed_set.add(current_station)

        for neighbor in current_station.links:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_station] + Euclidean(current_station, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_station
                g_score[neighbor] = tentative_g_score
                heappush(open_set, (tentative_g_score, neighbor))

    return [], 0.0

#* Uniform Cost Search algorithm
def ucs(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> Tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = [(0, start_station)]
    closed_set = set()
    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    came_from = {}

    start_time = time.time()
    while open_set:
        current_cost, current_station = heappop(open_set)

        if current_station == end_station:
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            end_time = time.time()
            return path, end_time - start_time

        closed_set.add(current_station)

        for neighbor in current_station.links:
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current_station] + 1  # Assuming all edges have a uniform cost of 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_station
                g_score[neighbor] = tentative_g_score
                heappush(open_set, (tentative_g_score, neighbor))

    return [], 0.0


#* main function
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


    #* Dijkstra's Algorithm
    path_dijkstra, time_dijkstra = dijkstra(start_station_name, end_station_name, stations)
    print("Dijkstra's Algorithm:")
    print("Shortest path:", path_dijkstra)
    print("Time taken:", time_dijkstra, "seconds\n")
    
    path,time = A_star(start_station_name, end_station_name, stations)
    print("Time taken:", time, "seconds\n")

    #* visualization the path
    plot_path(path, r'C:\Users\34071\PythonProjects\Artificial intelligience\STA303-Assignment03\visualization_underground\my_path_in_London_railway.html', stations, underground_lines)
