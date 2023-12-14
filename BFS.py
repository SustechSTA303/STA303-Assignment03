from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import Queue
import math


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
    parent = {} #记录节点的父节点
    open_list = Queue()
    go = {}
    path = []

    open_list.put(start_station.name)
    while(not open_list.empty()):
        now = open_list.get()
        if (now == end_station.name):
            break
        else:
            go[now] = True
            for adj in map[now].links:
                if((adj.name not in go)):
                    parent[adj.name] = now
                    open_list.put(adj.name)

    temp = ""
    path.append(end_station.name)
    while (not temp == start_station.name):
        temp = parent[path[-1]]
        path.append(temp)
    
    result = path[::-1]
    # # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    # print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    # print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    

    return result
