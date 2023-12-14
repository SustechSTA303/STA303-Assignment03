from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import queue
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
    open_list = [] #应当遍历的表
    go = {}
    path = []

    open_list.append(start_station.name)

    while (not (len(open_list) == 0)):
        now = open_list.pop()
        if (now == end_station.name):
            break
        else:
            go[now] = True
            for adj in map[now].links:
                if((adj.name not in go)):
                    parent[adj.name] = now
                    open_list.append(adj.name)
    temp = ""
    path.append(end_station.name)
    while (not temp == start_station.name):
        temp = parent[path[-1]]
        path.append(temp)
    
    result = path[::-1]    

    return result
