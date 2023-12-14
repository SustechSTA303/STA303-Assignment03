from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse

import math
from math import radians, sin, cos, sqrt, atan2, asin
from queue import PriorityQueue
import time

def cost(station1, station2):
    
    lat1 = radians(station1.position[0])
    lon1 = radians(station1.position[1])
    lat2 = radians(station2.position[0])
    lon2 = radians(station2.position[1])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    radius_earth_km = 6371.0
    
    distance = radius_earth_km * c
    
    return distance

def total_cost(open_station):
    total_cost = 0
    while open_station.parent != set():
        total_cost = total_cost + cost(open_station, open_station.parent)
        open_station = open_station.parent
    return total_cost

# # get_path of A star
# def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
#     """
#     runs astar on the map, find the shortest path between a and b
#     Args:
#         start_station_name(str): The name of the starting station
#         end_station_name(str): The name of the ending station
#         map(dict[str, Station]): Mapping between station names and station objects of the name,
#                                  Please refer to the relevant comments in the build_data.py
#                                  for the description of the Station class
#     Returns:
#         List[Station]: A path composed of a series of station_name
#     """
#     length = 0
    
#     # define the open set of cost + heuristic and station, closed set of station names, and open plus closed set of station names
#     open_stations = PriorityQueue()
#     closed_station_names = set()
#     open_closed_station_names = set()
    
#     # initialize home and destination
#     start_station = map[start_station_name]
#     end_station = map[end_station_name]
    
#     # initialize open set
#     open_stations.put((0 + heuristic(start_station, end_station), start_station))
    
#     while open_stations:
#         # get the station with lowest cost + heuristic in open set
#         f, current_station = open_stations.get()
#         current_cost = f - heuristic(current_station, end_station)
        
#         # return the path if the station is the destination
#         if current_station.name == end_station_name:
#             path = []
#             while current_station:
#                 path.append(current_station.name)
                
#                 if current_station.parent != set():
#                     length += cost(current_station, current_station.parent)
                    
#                 current_station = current_station.parent
#             print(length)
#             return list(reversed(path))
        
#         # add the station name into closed set and open plus closed set
#         closed_station_names.add(current_station.name)
#         open_closed_station_names.add(current_station.name)
        
#         # operate the station's neighbors
#         for neighbor in current_station.links:
#             if neighbor.name in closed_station_names:
#                 # the neighbor is in closed set, ignore
#                 continue
#             else:
#                 # check whether the neighbor is already in open set
#                 if neighbor.name in open_closed_station_names:
#                     # the station is a old station, save its old parent for comparison
#                     neighbor_parent = neighbor.parent 
#                     # check whether the new path to the neighbor is better than the old path to the neighbor 
#                     if (current_cost + cost(current_station, neighbor)) >= (total_cost(neighbor_parent) + cost(neighbor_parent, neighbor)):
#                         # the old path is better, ignore
#                         continue
#                     else:
#                         # the new path is better, remedy the neighbor's parent and put the neighbor with new cost into open set
#                         neighbor.parent = current_station
#                         open_stations.put((current_cost + cost(current_station, neighbor) + heuristic(neighbor, end_station), neighbor))
#                 # the neighbor is a new station, add it into open set
#                 else:
#                     neighbor.parent = current_station
#                     open_stations.put((current_cost + cost(current_station, neighbor) + heuristic(neighbor, end_station), neighbor))
#                     open_closed_station_names.add(neighbor.name)
            
#     return None         

# def heuristic(station1, station2):
#     lat1 = radians(station1.position[0])
#     lon1 = radians(station1.position[1])
#     lat2 = radians(station2.position[0])
#     lon2 = radians(station2.position[1])
    
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     R = 6371.0
    
# #     # great circle distance
# #     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
# #     c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
# #     distance = R * c
    
# #     # Euclidean distance
# #     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
# #     c = 2 * asin(sqrt(a))
    
# #     distance = R * c
    
#     # Manhattan distance
#     distance = R * (abs(dlon) + abs(dlat)) 
    
#     return distance


# # get_path of flawed UCS
# def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    
#     start_station = map[start_station_name]
#     end_station = map[end_station_name]
    
#     queue = PriorityQueue()
#     queue.put((0, [start_station_name]))
    
#     while not queue.empty():
#         current_cost, path = queue.get()
            
#         if end_station_name == path[-1]:
#             print(current_cost)
#             return path
            
#         for neighbor in map[path[-1]].links:
#             if neighbor.name in path:
#                 continue
#             else:
#                 new_cost = current_cost + cost(map[path[-1]], neighbor)
#                 new_path = path + [neighbor.name]
#                 queue.put((new_cost, new_path))
                
#     return None

# get_path of dijkstra
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    length = 0
    
     # define the open set of total cost and station, closed set of station names, and open plus closed set of station names
    open_stations = PriorityQueue()
    closed_station_names = set()
    open_closed_station_names = set()
    
    # initialize home and destination
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    # initialize open set
    open_stations.put((0, start_station))
    
    
    while open_stations:
        # get the station with lowest total cost in open set
        current_total_cost, current_station = open_stations.get()
        
        # return the path if the station is the destination
        if current_station.name == end_station_name:
            path = []
            while current_station:
                path.append(current_station.name)
                
                if current_station.parent != set():
                    length += cost(current_station, current_station.parent)
                    
                current_station = current_station.parent
            print(length)
            return list(reversed(path))
        
        # add the station name into closed set and open plus closed set
        closed_station_names.add(current_station.name)
        open_closed_station_names.add(current_station.name)
        
        # operate the station's neighbors
        for neighbor in current_station.links:
            
            if neighbor in closed_station_names:
                # the neighbor is in closed set, ignore
                continue
            else:
                # check whether the neighbor is already in open set
                if neighbor.name in open_closed_station_names:
                    # the station is a old station, save its old parent for comparison
                    neighbor_parent = neighbor.parent 
                    # check whether the new path to the neighbor is better than the old path to the neighbor
                    if (total_cost(current_station) + cost(current_station, neighbor)) >= total_cost(neighbor):
                        # the old path is better, ignore
                        continue
                    else:
                        # the new path is better, remedy the neighbor's parent and put the neighbor with new total cost into open set
                        neighbor.parent = current_station
                        open_stations.put((total_cost(neighbor), neighbor))
                # the neighbor is a new station, add it into open set
                else:
                    neighbor.parent = current_station
                    open_stations.put((total_cost(neighbor), neighbor))
                    open_closed_station_names.add(neighbor.name)
                    
    return None


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
    
    time_start = time.time()
    path = get_path(start_station_name, end_station_name, stations)
    time_end = time.time()
    
    print(time_end - time_start)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
