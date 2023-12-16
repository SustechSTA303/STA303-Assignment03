from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
import heapq
from queue import PriorityQueue
import time

counter = 0
times =10
#heuristic of SqareRoot

def calculate_total_cost(path: List[str], map: dict[str, Station],heuristic) -> float:
    total_cost = 0.0
    for i in range(len(path) - 1):
        station_name1 = path[i]
        station_name2 = path[i + 1]
        station1 = map[station_name1]
        station2 = map[station_name2]
        total_cost += heuristic(station1, station2)
    return total_cost

def heuristic_sqrt(station1, station2):
    lat1, long1 = station1.position
    lat2, long2 = station2.position
    return math.sqrt((lat2 - lat1)**2 + (long2 - long1)**2)

def heuristic_Manhattan(station1, station2):
    lat1, long1 = station1.position
    lat2, long2 = station2.position
    return abs(lat2 - lat1) + abs(long2 - long1)

def heuristic_1(station1, station2):
    return 1

def haversine_distance(station1, station2):
    lat1, long1 = station1.position
    lat2, long2 = station2.position
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, long1, lat2, long2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_earth = 6371  # Radius of the Earth in kilometers
    distance = radius_earth * c
    return distance

heuristic = heuristic_Manhattan
def ucs(graph, home, destination, map):
    """
    Perform Uniform Cost Search on a graph from a start location (home) to a goal location (destination).

    Parameters:
    graph (dict): A dictionary representation of the graph where keys are location names and values
                  are lists of neighbors.
    home (str): The starting location in the graph.
    destination (str): The goal location to reach in the graph.
    map (dict): Dictionary mapping station names to Station objects

    Returns:
    tuple: result(tuple) : A tuple containing the total cost (int) and path as a list of locations (str) from 'home' to 'destination'.
    """

    
    if home not in graph:
        raise TypeError(str(home) + ' not found in graph!')
    if destination not in graph:
        raise TypeError(str(destination) + ' not found in graph!')

    queue = PriorityQueue()
    queue.put((0, [home]))
    visited = set()

    while not queue.empty():
        now = queue.get()
        nowcost = now[0]  # Current node's cost
        nowPath = now[1]  # Current node's path

        if nowPath[-1] == destination:

            return nowPath, nowcost  # If the current node is the destination, return the path and cost

        if nowPath[-1] in visited:
            continue  # If the current node has been visited, skip it

        visited.add(nowPath[-1])

        for neighbor_station in graph[nowPath[-1]]:
            if neighbor_station not in visited:
                neighbor = map[neighbor_station]
                appendedPath = nowPath[:]  # Clone the old path for the new one
                appendedPath.append(neighbor_station)
                # Calculate the cost using the haversine distance
                new_cost = nowcost + heuristic(map[nowPath[-1]], neighbor)
                queue.put((new_cost, appendedPath))  # Put it into the queue with updated cost

    # If the goal is not reachable, return None
    return None, None

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> list[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Convert station links to station names for the graph representation
    graph = {station.name: [neighbor.name for neighbor in station.links] for station in map.values()}

    # Perform UCS
    path, cost = ucs(graph, start_station.name, end_station.name, map)
    
    if path:
        return path  # Return the path from UCS
    
    return []  # Return an empty list if no path is found

if __name__ == '__main__':

#     # 创建ArgumentParser对象
#     parser = argparse.ArgumentParser()
#     # 添加命令行参数
#     parser.add_argument('start_station_name', type=str, help='start_station_name')
#     parser.add_argument('end_station_name', type=str, help='end_station_name')
#     args = parser.parse_args()
#     start_station_name = args.start_station_name
#     end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    start_station_name = "Wimbledon"
    end_station_name = "Seven Sisters"
    # Record the start time
    start_time = time.time()

    # Your Python code goes here
    for i in range(10):
        counter +=1
        path = get_path(start_station_name, end_station_name, stations)
        if(counter == times):
            with open('Final_Path_length.txt', 'a') as file:
                    file.write(f"UCS heuristic_sqrt: {calculate_total_cost(path,stations,heuristic):.6f}\n")

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    with open('Final_Path_length.txt', 'a') as file:
                    file.write(f"time:{elapsed_time:.6f}\n")
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
