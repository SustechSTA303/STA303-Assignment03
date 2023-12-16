from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
import heapq
import time

counter =0
times = 10

class Node:
    def __init__(self, station, cost):
        self.station = station
        self.cost = cost
    
    def __lt__(self, other):
        return self.cost < other.cost

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

heuristic_functions = {
    heuristic_sqrt: 'heuristic_sqrt',
    heuristic_Manhattan: 'heuristic_Manhattan',
    heuristic_1: 'heuristic_1',
    haversine_distance: 'haversine_distance'
}

###########################
heuristic = heuristic_sqrt
###########################

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    visited = set()
    queue = []
    heapq.heappush(queue, Node(start_station, 0))
    parent = {start_station: None}
    cost_so_far = {start_station: 0}
    
    while queue:
        current_node = heapq.heappop(queue)
        current_station = current_node.station
        
        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = parent[current_station]
            return path[::-1]
        
        visited.add(current_station)
        
        for neighbor_station in current_station.links:
            neighbor = map[neighbor_station.name]
            new_cost = cost_so_far[current_station] + heuristic(current_station, neighbor)  # Modify this line if you have specific costs between stations
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(queue, Node(neighbor, new_cost))
                parent[neighbor] = current_station
    
    return []

if __name__ == '__main__':
    with open('Dijkstra.txt', 'a') as file:
        file.write(f"Heuristic: {heuristic_functions.get(heuristic, 'Unknown heuristic')}\n")
        file.write(f"*********************************\n")
    
    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    station_pairs = [
("Paddington","Shepherd's Bush (C)"),
("East Acton","High Barnet"),
("Farringdon","Edgware"),
("South Kensington","Tower Gateway"),
("St. John's Wood","Park Royal"),
("Sudbury Town","Harrow & Wealdston"),
("Blackhorse Road","Roding Valley"),
("Notting Hill Gate","Holborn"),
("South Ealing","Clapham South"),
("Royal Victoria","Croxley"),
("Marble Arch","West Finchley"),
("Surrey Quays","Hammersmith"),
("West Harrow","Borough"),
("Latimer Road","Chesham"),
("Kilburn Park","Shadwell"),
("Queen's Park","Wembley Central"),
("Latimer Road","Snaresbrook"),
("Park Royal","Colliers Wood"),
("East India","Perivale"),
("Hainault","Temple")
    # Add more station pairs here
    ]
    pair_num = len(station_pairs)
    pair_count = 0
    time_total=0
    for start, end in station_pairs:
        start_station_name = start
        end_station_name = end
        pair_count += 1
        start_time = time.time()
        # Record the start time
        path = get_path(start_station_name, end_station_name, stations)

        # Your Python code goes here
        end_time = time.time()

        # Calculate the elapsed time
        time_total += end_time - start_time
        if(pair_count == pair_num):
            with open('Dijkstra.txt', 'a') as file:
#                     file.write(f"A*: {path_total/pair_num:.6f}\n")
                    file.write(f"time:{time_total:.6f}\n")
                    file.write(f"---------------------------------\n")
    # Record the end time

    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
