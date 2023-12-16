from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
import heapq
import time

counter = 0
times = 10

def calculate_total_cost(path: List[str], map: dict[str, Station],heuristic) -> float:
    total_cost = 0.0
    for i in range(len(path) - 1):
        station_name1 = path[i]
        station_name2 = path[i + 1]
        station1 = map[station_name1]
        station2 = map[station_name2]
        total_cost += heuristic(station1, station2)
    return total_cost

class Node:
    def __init__(self, station, cost_so_far, estimated_total_cost):
        self.station = station
        self.cost_so_far = cost_so_far
        self.estimated_total_cost = estimated_total_cost
    
    def __lt__(self, other):
        return self.estimated_total_cost < other.estimated_total_cost

#heuristic of SqareRoot
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

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    heuristic = heuristic_Manhattan  # Change the heuristic function as needed

    visited = set()
    queue = []
    heapq.heappush(queue, Node(start_station, 0, heuristic(start_station, end_station)))
    parent = {start_station: None}

    while queue:
        current_node = heapq.heappop(queue)
        current_station = current_node.station

        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = parent[current_station]
            if(counter == times):
                with open('Final_Path_length.txt', 'a') as file:
                    file.write(f"Greedy: {calculate_total_cost(path,map,heuristic):.6f}\n")
            return path[::-1]

        visited.add(current_station)

        for neighbor_station in current_station.links:
            neighbor = map[neighbor_station.name]
            if neighbor not in visited:
                total_cost = heuristic(neighbor, end_station)
                heapq.heappush(queue, Node(neighbor, 0, total_cost))
                parent[neighbor] = current_station

    return []


if __name__ == '__main__':

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    start_station_name = "Wimbledon"
    end_station_name = "Seven Sisters"
    # Record the start time
    start_time = time.time()

    for i in range(10):
        counter +=1
        path = get_path(start_station_name, end_station_name, stations)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    with open('Final_Path_length.txt', 'a') as file:
                    file.write(f"time:{elapsed_time:.6f}\n")
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
