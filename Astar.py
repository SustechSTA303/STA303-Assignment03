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

    open_list = queue.PriorityQueue() #用于排序进行
    open_record = list() #记录开表里面的node
    open_cost = {} # 记录在开表里面每个的nodecost

    close_list =list() #记录闭表的node的名字

    parent = {} # 用于记录每个close里面node的最新父节点
    finish = False
    path = []

    open_list.put((0,start_station.name)) #先将起点加入
    open_record.append(start_station.name)
    open_cost[start_station.name] = 0


    while not open_list.empty() :
        now = open_list.get()[1]
        now = map[now]
        open_record.remove(now.name) #去除记录里面的
        if(now.name == end_station.name):
            close_list.append(end_station.name)
            finish = True
            break
        else:
            close_list.append(now.name)
            for adj in now.links:
                if(adj.name not in close_list):
                    if((adj.name not in open_record)):
                        cost = open_cost[now.name] + distance(adj.position,now.position) 
                        F = float(cost + distance(adj.position,end_station.position))
                        open_list.put((F,adj.name))
                        open_record.append(adj.name)
                        open_cost[adj.name] = cost
                        parent[adj.name] = now.name
                    elif((adj.name in open_record)):
                        cost = open_cost[now.name] + distance(now.position,adj.position)
                        if(cost <= open_cost[adj.name]):
                            F = float(cost + distance(adj.position,end_station.position))
                            open_list.put((F,adj.name))
                            open_record.append(adj.name)
                            open_cost[adj.name] = cost
                            parent[adj.name] = now.name
    #追溯路径
    if(finish):
        temp = ""
        path.append(end_station.name)
        while (not temp == start_station.name):
            temp = parent[path[-1]]
            path.append(temp)

    path = path[::-1]
    return path


# def distance(ori,des):
#     dist = math.sqrt((ori[0]-des[0])**2 + (ori[1]-des[1])**2)
#     return dist
def distance(ori,des):
    dist = float(abs(ori[0]-des[0])+ abs(ori[1]-des[1]))
    return dist
