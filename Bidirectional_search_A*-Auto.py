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
# (Your previous code remains unchanged here)

# Set up the heuristics and data structures for the start and goal searches
heuristic = heuristic_Manhattan
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]


    
    # Data structures for start to end search
    start_visited = set()
    start_queue = []
    heapq.heappush(start_queue, Node(start_station, 0, heuristic(start_station, end_station)))
    start_parent = {start_station: None}
    start_cost_so_far = {start_station: 0}

    # Data structures for end to start search
    end_visited = set()
    end_queue = []
    heapq.heappush(end_queue, Node(end_station, 0, heuristic(end_station, start_station)))
    end_parent = {end_station: None}
    end_cost_so_far = {end_station: 0}

    while start_queue and end_queue:
        # Expand start search
        current_start_node = heapq.heappop(start_queue)
        current_start_station = current_start_node.station

        start_visited.add(current_start_station)

        for neighbor_station in current_start_station.links:
            neighbor = map[neighbor_station.name]
            new_start_cost = start_cost_so_far[current_start_station] + heuristic(current_start_station, neighbor)
            if neighbor not in start_cost_so_far or new_start_cost < start_cost_so_far[neighbor]:
                start_cost_so_far[neighbor] = new_start_cost
                total_start_cost = new_start_cost + heuristic(neighbor, end_station)
                heapq.heappush(start_queue, Node(neighbor, new_start_cost, total_start_cost))
                start_parent[neighbor] = current_start_station

                # Check if a common node is found
                if neighbor in end_visited:
                    merged_path = _merge_paths(start_parent, end_parent, neighbor)
                    
                    return merged_path

        # Expand end search
        current_end_node = heapq.heappop(end_queue)
        current_end_station = current_end_node.station

        end_visited.add(current_end_station)

        for neighbor_station in current_end_station.links:
            neighbor = map[neighbor_station.name]
            new_end_cost = end_cost_so_far[current_end_station] + heuristic(current_end_station, neighbor)
            if neighbor not in end_cost_so_far or new_end_cost < end_cost_so_far[neighbor]:
                end_cost_so_far[neighbor] = new_end_cost
                total_end_cost = new_end_cost + heuristic(neighbor, start_station)
                heapq.heappush(end_queue, Node(neighbor, new_end_cost, total_end_cost))
                end_parent[neighbor] = current_end_station

                # Check if a common node is found
                if neighbor in start_visited:
                    merged_path = _merge_paths(start_parent, end_parent, neighbor)
                    
                    return merged_path

    # Merge paths if a common node is found
    common_node = _find_common_node(start_visited, end_visited)
    if common_node:
        return _merge_paths(start_parent, end_parent, common_node)

    # If no common node found and both searches have ended without meeting,
    # attempt to connect the paths manually
    return _connect_paths(start_parent, end_parent, start_visited, end_visited)

def _connect_paths(start_parent, end_parent, start_visited, end_visited):
    # Attempt to connect paths manually
    for node in start_visited:
        if node in end_visited:
            return _merge_paths(start_parent, end_parent, node)

    # If no common node found, return an empty path
    return []

# Rest of your code (heuristic functions, Node class, etc.) remains unchanged

def _merge_paths_meta(start_parent,end_parent,  intersection_node):
    # Merge paths from start and end searches
    start_path = []
    end_path = []

    # Traverse from intersection node to start node
    while intersection_node:
        start_path.append(intersection_node.name)
        intersection_node = start_parent.get(intersection_node)  # Using .get() to handle None

    # Reverse and store the start path
    start_path = start_path[::-1]

    # Reset intersection_node to the common node
    intersection_node = end_parent.get(intersection_node)

    # Traverse from intersection node to end node
    while intersection_node:
        end_path.append(intersection_node.name)
        intersection_node = end_parent.get(intersection_node)

    # Remove the common node (intersection node) if it appears in the end path
    end_path = end_path[:-1]

    return start_path + end_path[::-1]

def _merge_paths(start_parent,end_parent, intersection_node):
    start_parent_copy = start_parent.copy()
    end_parent_copy = end_parent.copy()
    a = _merge_paths_meta(start_parent,end_parent, intersection_node)[::-1]
    a.pop(0)
    b=_merge_paths_meta(end_parent,start_parent, intersection_node)
#     b.pop(0)
    return b+a



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
    for i in range(times):
        counter+=1
        path = get_path(start_station_name, end_station_name, stations)
        
        if(counter == times):
                        with open('Final_Path_length.txt', 'a') as file:
                            file.write(f"A* bidirectional heuristic_sqrt2:{calculate_total_cost(path,stations,heuristic)}\n")

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    with open('Final_Path_length.txt', 'a') as file:
                    file.write(f"time:{elapsed_time:.6f}\n")
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
