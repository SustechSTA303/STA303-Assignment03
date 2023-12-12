import argparse
from tqdm import tqdm
import pandas as pd
from queue import PriorityQueue, Queue
import time
from typing import List, Tuple

import warnings

warnings.filterwarnings("ignore")

from plot_underground_path import plot_path
from build_data import Station, build_data
from functions import heuristics, cost, pathLength, random_choice

"""
runs a star on the map, find the shortest path between a and b
Args:
    start_station_name(str): The name of the starting station
    end_station_name(str): str The name of the ending station
    map(dict[str, Station]): Mapping between station names and station objects of the name,
Please refer to the relevant comments in the build_data.py for the description of the Station class
Returns:
    List[Station]: A path composed of a series of station_name

# You can obtain the Station objects of the starting and ending station through the following code

# Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
# print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
# print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

"""


# * BFS algorithm
def BFS(
    start_station_name: str, end_station_name: str, map: dict[str, Station]
) -> Tuple[List[str], float, float]:
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
            path_length = pathLength(path, map)
            return path, path_length, end_time - start_time

        for next in current.links:
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    return [], 0, 0.0


# * Dijkstra algorithm
def Dijkstra(
    start_station_name: str,
    end_station_name: str,
    map: dict[str, Station],
    cost_type: str,
) -> Tuple[List[str], float, float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    start_time = time.time()
    while not frontier.empty():
        tmp, current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            path_length = pathLength(path, map)
            return path, path_length, end_time - start_time

        for next in current.links:
            new_cost = cost_so_far[current] + cost(current, next, cost_type)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put((priority, next))
                came_from[next] = current
    return [], 0, 0.0


# * Bellman-Ford algorithm
def BellmanFord(
    start_station_name: str,
    end_station_name: str,
    map: dict[str, Station],
    cost_type: str,
) -> Tuple[List[str], float, float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    came_from = dict()
    cost_so_far = dict()

    for station in map.values():
        cost_so_far[station] = float("inf")

    cost_so_far[start_station] = 0

    start_time = time.time()

    for _ in range(len(map) - 1):
        for current in map.values():
            for next_station in current.links:
                new_cost = cost_so_far[current] + cost(current, next_station, cost_type)
                if new_cost < cost_so_far[next_station]:
                    cost_so_far[next_station] = new_cost
                    came_from[next_station] = current

    end_time = time.time()

    # Check for negative cycles
    for current in map.values():
        for next_station in current.links:
            if (
                cost_so_far[current] + cost(current, next_station, cost_type)
                < cost_so_far[next_station]
            ):
                # Negative cycle detected
                return [], float("-inf"), end_time - start_time

    # Reconstruct path
    path = []
    current = end_station
    while current:
        path.insert(0, current.name)
        current = came_from.get(current)

    path_length = cost_so_far[end_station]

    return path, path_length, end_time - start_time


# * Greedy BFS algorithm
def G_BFS(
    start_station_name: str,
    end_station_name: str,
    map: dict[str, Station],
    heuristic_type: str,
    heuristic_weight: float = 1.0,
) -> Tuple[List[str], float, float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    start_time = time.time()
    while not frontier.empty():
        tmp, current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            path_length = pathLength(path, map)
            return path, path_length, end_time - start_time

        for next in current.links:
            if next not in came_from:
                priority = heuristics(
                    end_station, next, heuristic_type, heuristic_weight
                )
                frontier.put((priority, next))
                came_from[next] = current
    return [], 0, 0.0


# * A star algorithm
def A_star(
    start_station_name: str,
    end_station_name: str,
    map: dict[str, Station],
    cost_type: str,
    heuristic_type: str,
    heuristic_weight: float = 1.0,
) -> Tuple[List[str], float, float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    start_time = time.time()
    while not frontier.empty():
        tmp, current = frontier.get()

        if current == end_station:
            end_time = time.time()
            path = []
            while current:
                path.insert(0, current.name)
                current = came_from.get(current)
            path_length = pathLength(path, map)
            return path, path_length, end_time - start_time

        for next in current.links:
            new_cost = cost_so_far[current] + cost(current, next, cost_type)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristics(
                    end_station, next, heuristic_type, heuristic_weight
                )
                frontier.put((priority, next))
                came_from[next] = current
    return [], 0, 0.0


# * Bi-directional A star algorithm
def bi_directional_A_star(
    start_station_name: str,
    end_station_name: str,
    map: dict[str, Station],
    cost_type: str,
    heuristic_type: str,
    heuristic_weight: float = 1.0,
) -> Tuple[List[str], float, float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    forward_frontier = PriorityQueue()
    backward_frontier = PriorityQueue()

    forward_frontier.put((0, start_station))
    backward_frontier.put((0, end_station))

    forward_came_from = {start_station: None}
    backward_came_from = {end_station: None}

    forward_cost_so_far = {start_station: 0}
    backward_cost_so_far = {end_station: 0}

    intersection_station = None
    min_cost = float("inf")

    end_time = 0.0
    start_time = time.time()
    while not forward_frontier.empty() and not backward_frontier.empty():
        forward_tmp, forward_current = forward_frontier.get()
        backward_tmp, backward_current = backward_frontier.get()

        # Check if paths meet
        if (
            forward_current in backward_cost_so_far
            and backward_cost_so_far[forward_current]
            + forward_cost_so_far[forward_current]
            < min_cost
        ):
            min_cost = (
                backward_cost_so_far[forward_current]
                + forward_cost_so_far[forward_current]
            )
            intersection_station = forward_current
            end_time = time.time()

        # Expand forward
        for forward_next in forward_current.links:
            new_forward_cost = forward_cost_so_far[forward_current] + cost(
                forward_current, forward_next, cost_type
            )
            if (
                forward_next not in forward_cost_so_far
                or new_forward_cost < forward_cost_so_far[forward_next]
            ):
                forward_cost_so_far[forward_next] = new_forward_cost
                forward_priority = new_forward_cost + heuristics(
                    end_station, forward_next, heuristic_type, heuristic_weight
                )
                forward_frontier.put((forward_priority, forward_next))
                forward_came_from[forward_next] = forward_current

        # Expand backward
        for backward_next in backward_current.links:
            new_backward_cost = backward_cost_so_far[backward_current] + cost(
                backward_current, backward_next, cost_type
            )
            if (
                backward_next not in backward_cost_so_far
                or new_backward_cost < backward_cost_so_far[backward_next]
            ):
                backward_cost_so_far[backward_next] = new_backward_cost
                backward_priority = new_backward_cost + heuristics(
                    start_station, backward_next, heuristic_type, heuristic_weight
                )
                backward_frontier.put((backward_priority, backward_next))
                backward_came_from[backward_next] = backward_current

    # Reconstruct path
    path = []
    current = intersection_station
    while current:
        path.insert(0, current.name)
        current = forward_came_from.get(current)

    current = backward_came_from.get(intersection_station)
    while current:
        path.append(current.name)
        current = backward_came_from.get(current)

    path_length = pathLength(path, map)
    return path, path_length, end_time - start_time


# *test_time
def test_time(Station_pair: list, map: dict[str, Station]):
    df = pd.DataFrame()
    for start, end in tqdm(Station_pair):
        result_dict = {
            "BFS": BFS(start, end, map)[2]*1000,
            "Dijkstra(1)": Dijkstra(start, end, map, cost_type="1")[2]*1000,
            "Dijkstra(Haversine)": Dijkstra(start, end, map, cost_type="Haversine")[2]*1000,
            "Dijkstra(Manhattan)": Dijkstra(start, end, map, cost_type="Manhattan")[2]*1000,
            "Dijkstra(Euclidean)": Dijkstra(start, end, map, cost_type="Euclidean")[2]*1000,
            "BellmanFord(1)": BellmanFord(start, end, map, cost_type="1")[2]*1000,
            # "BellmanFord(Haversine)": BellmanFord(
            #     start, end, map, cost_type="Haversine"
            # )[2]*1000,
            # "BellmanFord(Manhattan)": BellmanFord(
            #     start, end, map, cost_type="Manhattan"
            # )[2]*1000,
            "BellmanFord(Euclidean)": BellmanFord(
                start, end, map, cost_type="Euclidean"
            )[2],
            "G_BFS(Haversine)": G_BFS(start, end, map, heuristic_type="Haversine")[2]*1000,
            "G_BFS(Manhattan)": G_BFS(start, end, map, heuristic_type="Manhattan")[2]*1000,
            "G_BFS(Euclidean)": G_BFS(start, end, map, heuristic_type="Euclidean")[2]*1000,
            "A_star(Euclidean, Euclidean)": A_star(
                start, end, map, cost_type="Euclidean", heuristic_type="Euclidean"
            )[2]*1000,
            "bi_directional_A_star(Euclidean,Euclidean)": bi_directional_A_star(
                start, end, map, cost_type="Euclidean", heuristic_type="Euclidean"
            )[2]*1000,
        }

        # Append the dictionary to the DataFrame
        df = df.append(result_dict, ignore_index=True)
    df.to_csv("output/time.csv")


# *test_pathlength
def test_pathlength(map: dict[str, Station]):
    station_pairs = (
        ("Cockfosters", "Cockfosters"),
        ("West Ruislip", "Epping"),
        ("Morden", "High Barnet"),
        ("Upminster", "Richmond"),
        ("Brixton", "Walthamstow"),
    )
    df = pd.DataFrame()
    for start, end in station_pairs:
        


# * main function
if __name__ == "__main__":
    # 创建ArgumentParser对象
    # parser = argparse.ArgumentParser()
    # # 添加命令行参数
    # parser.add_argument('start_station_name', type=str, help='start_station_name')
    # parser.add_argument('end_station_name', type=str, help='end_station_name')
    # args = parser.parse_args()
    # start_station_name = args.start_station_name
    # end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()

    start_station_name = "Archway"
    end_station_name = "Bermondsey"

    # path,path_length,time_ = BFS(start_station_name, end_station_name, stations)
    # # path,path_length,time_ = A_star(start_station_name, end_station_name, stations, cost_type='Euclidean', heuristic_type='Euclidean', heuristic_weight=1.0)
    # path,path_length,time_ = Dijkstra(start_station_name, end_station_name, stations, cost_type='Euclidean')
    # path,path_length,time_ = Dijkstra(start_station_name, end_station_name, stations, cost_type='Euclidean')
    # path, path_length, time_ = BellmanFord(start_station_name, end_station_name, stations, cost_type="Euclidean")
    # path,path_length,time_ = G_BFS(start_station_name, end_station_name, stations, heuristic_type='Euclidean', heuristic_weight=1.0)
    # path,path_length,time_ = bi_directional_A_star(start_station_name, end_station_name, stations, cost_type='Euclidean', heuristic_type='Euclidean', heuristic_weight=1.0)
    # print("Time taken:", time_ * 1000, "ms\n")
    # print("Path length:", path_length, "kilometers\n")



    ##* test time of algorithm
    ## randomly choose 150 pairs of station
    Station_pair = random_choice(stations.keys(), 250)
    test_time(Station_pair, stations)

    ##* test path_length of algorithm

    # #* visualization the path
    # plot_path(path, r'C:\Users\34071\PythonProjects\Artificial intelligience\STA303-Assignment03\visualization_underground\my_path_in_London_railway.html', stations, underground_lines)
