from queue import PriorityQueue, Queue
import time
from typing import List, Tuple

from plot_underground_path import plot_path
from build_data import Station, build_data
from cost_heuristics_function import heuristics,cost
import argparse

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

# Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
# print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
# print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

#* BFS algorithm
def BFS(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> Tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = Queue()
    frontier.put(start_station)
    came_from = dict()
    came_from[start_station] = None

    start_time = time.time()
    while not frontier.empty():
        current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            return path, end_time-start_time

        for next in current.links:
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    return [],0.0

#* UCS algorithm
def UCS(start_station_name: str, end_station_name: str, map: dict[str, Station], cost_type:str) -> Tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue()
    frontier.put((0,start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    start_time = time.time()
    while not frontier.empty():
        tmp,current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            return path, end_time-start_time

        for next in current.links:
            new_cost = cost_so_far[current] + cost(current, next, cost_type)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put((priority,next))
                came_from[next] = current
    return [],0.0

#* Greedy BFS algorithm
def G_BFS(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic_type:str, heuristic_weight: float=1.0) -> Tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue()
    frontier.put((0,start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    start_time = time.time()
    while not frontier.empty():
        tmp,current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            return path, end_time-start_time

        for next in current.links:
            if next not in came_from:
                priority = heuristics(end_station, next, heuristic_type,heuristic_weight)
                frontier.put((priority,next))
                came_from[next] = current
    return [],0.0

#* A star algorithm
def A_star(start_station_name: str, end_station_name: str, map: dict[str, Station], cost_type:str, heuristic_type:str, heuristic_weight: float=1.0) -> Tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue()
    frontier.put((0,start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    start_time = time.time()
    while not frontier.empty():
        tmp,current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            return path, end_time-start_time

        for next in current.links:
            new_cost = cost_so_far[current] + cost(current, next, cost_type)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristics(end_station, next, heuristic_type,heuristic_weight)
                frontier.put((priority,next))
                came_from[next] = current
    return [],0.0

#* main function
if __name__ == '__main__':

    # # 创建ArgumentParser对象
    # parser = argparse.ArgumentParser()
    # # 添加命令行参数
    # parser.add_argument('start_station_name', type=str, help='start_station_name')
    # parser.add_argument('end_station_name', type=str, help='end_station_name')
    # args = parser.parse_args()
    # start_station_name = args.start_station_name
    # end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()

    start_station_name = "Chesham"
    end_station_name = "Hainault"

    path,time = BFS(start_station_name, end_station_name, stations)
    # path,time = A_star(start_station_name, end_station_name, stations, cost_type='Euclidean', heuristic_type='Euclidean', heuristic_weight=1.0)
    # path,time = UCS(start_station_name, end_station_name, stations, cost_type='Euclidean')
    # path,time = G_BFS(start_station_name, end_station_name, stations, heuristic_type='Euclidean', heuristic_weight=1.0)
    print("Time taken:", time, "seconds\n")

    #* visualization the path
    plot_path(path, r'C:\Users\34071\PythonProjects\Artificial intelligience\STA303-Assignment03\visualization_underground\my_path_in_London_railway.html', stations, underground_lines)
