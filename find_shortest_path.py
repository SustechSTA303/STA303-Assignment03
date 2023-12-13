from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time
import copy
import heapq
import random
import numpy as np
from queue import PriorityQueue, Queue

class CustomPriorityQueue():
    def __init__(self):
        self.queue = []
        self.counter = 0  # 用于保证每个元素的唯一性

    def put(self, item):
        entry = (item[0], self.counter, item[1])
        heapq.heappush(self.queue, entry)
        self.counter += 1

    def get(self):
        next_item=heapq.heappop(self.queue)
        return next_item[0],next_item[2]

    def empty(self):
        return not bool(self.queue)

# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
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
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')

    # if home not in graph:
    #     raise TypeError(str(home) + ' not found in graph!')
    # if destination not in graph:
    #     raise TypeError(str(destination) + ' not found in graph!')

    # distances = {node: float('inf') for node in stations.keys()}
    # predecessors = {node: None for node in stations.keys()}
    # distances[start_station_name] = 0

    # queue = PriorityQueue()
    # queue.put((0, start_station))

    # while not queue.empty():
    #     # print(queue.get())
    #     cur_item=queue.get()
    #     # print(cur_item[0])
    #     current_distance, current_node = queue.get()

    #     if current_distance > distances[current_node.name]:
    #         continue

    #     for station in current_node.links:
    #         distance = current_distance + np.linalg.norm(np.array(station.position)-np.array(current_node.position))
    #         if distance < distances[station.name]:
    #             distances[station.name] = distance
    #             predecessors[station.name] = current_node
    #             queue.put(distance, [station])
    
    # path = []
    # print(predecessors[end_station_name])
    # goal_node=copy.deepcopy(end_station)
    # while goal_node is not None:
    #     path.insert(0, goal_node.name)
    #     goal_node = predecessors[goal_node.name]
    # print(path)
    # return path

    queue = CustomPriorityQueue()
    # queue=PriorityQueue()
    queue.put((0, [start_station]))
    
    visited = set()
    # start_time=time.time()
    while not queue.empty():
        current_cost, current_path = queue.get()
        current_location = current_path[-1]

        if current_location.name == end_station.name:
            # end_time=time.time()
            # print(f"time consumed:{(end_time-start_time)*1000}")  
            return [station.name for station in current_path]

        if current_location not in visited:
            visited.add(current_location)

            for station in current_location.links:
                # new_cost = max(abs(a - b) for a, b in zip(station.position, end_station.position))
                # new_cost = sum(abs(a - b) for a, b in zip(station.position, end_station.position))
                # new_cost = np.linalg.norm(np.array(station.position)-np.array(end_station.position))
                # new_cost = current_cost + np.linalg.norm(np.array(station.position)-np.array(current_location.position))
                # new_cost = current_cost + np.linalg.norm(np.array(station.position)-np.array(current_location.position))-sum(abs(a - b) for a, b in zip(current_location.position, end_station.position))+sum(abs(a - b) for a, b in zip(station.position, end_station.position))
                new_cost=current_cost + np.linalg.norm(np.array(station.position)-np.array(current_location.position))-(max(abs(a - b) for a, b in zip(current_location.position, end_station.position))-max(abs(a - b) for a, b in zip(station.position, end_station.position)))
                # new_cost = current_cost + np.linalg.norm(np.array(station.position)-np.array(current_location.position))-np.linalg.norm(np.array(current_location.position)-np.array(end_station.position))+np.linalg.norm(np.array(station.position)-np.array(end_station.position))
                
                new_path = current_path + [station]
                queue.put((new_cost, new_path))
            ###############################################################
        # Your task: Complete the implementation here.
        # - Check if the current location is the goal (destination).
        # - Iterate over the neighbors of the current location.
        # - For each neighbor, calculate the total cost from the start location.
        # - Add the neighbor to the priority queue if it hasn't been visited  
    # If the goal is not reachable, return None
    return None
    pass

def pick_n_pairs_of_keys(dictionary, n):
    all_keys = list(dictionary.keys())
    
    selected_pairs = [(random.choice(all_keys), random.choice(all_keys)) for _ in range(n)]
    
    return selected_pairs

if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    # parser.add_argument('start_station_name', type=str, default="Wembley Park", help='start_station_name')
    # parser.add_argument('end_station_name', type=str, default="Gunnersbury", help='end_station_name')
    # args = parser.parse_args()
    # start_station_name = args.start_station_name
    # end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    # path = get_path(start_station_name, end_station_name, stations)
    # cost=sum([np.linalg.norm(np.array(stations[path[i]].position)-np.array(stations[path[i+1]].position)) for i in range(len(path)-1)])
    
    random.seed(42)
    selected_pairs = pick_n_pairs_of_keys(stations, 1000)
    cost=0
    total_time_start=time.time()
    for start_station_name, end_station_name in selected_pairs:
        path = get_path(start_station_name, end_station_name, stations)
        cost+=sum([np.linalg.norm(np.array(stations[path[i]].position)-np.array(stations[path[i+1]].position)) for i in range(len(path)-1)])
    total_time_end=time.time()
    print(f"total time: {total_time_end-total_time_start}")
    print("length:"+str(cost))
    # # visualization the path
    # # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    # plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
