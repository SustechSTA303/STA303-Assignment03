from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse

import time
import random
from queue import PriorityQueue
from queue import Queue
import csv
import os
from distance_functions import Euclidean_distance, Manhattan_distance, Diagonal_distance

csv_file_path = 'london/underground_stations.csv'  
column_name = 'name' 
column_data=[]
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    column_data = [row[column_name] for row in reader]


def get_path_As(start_station_name: str, end_station_name: str, map: dict[str, Station], distance_func=None) -> tuple[List[str], float]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0
    if distance_func is None:
        distance_func = Euclidean_distance
    while not frontier.empty():
        current = frontier.get()[1]
        if current == end_station:
            break
        for next in current.links:
            new_cost = cost_so_far[current] + distance_func(current.position[0], next.position[0],current.position[1], next.position[1])
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + distance_func(next.position[0], end_station.position[0],next.position[1], end_station.position[1])
                frontier.put((priority, next))
                came_from[next] = current
    path = []
    s = 0
    current = end_station
    while current != start_station:
        path.append(current.name)
        s += Euclidean_distance(current.position[0], came_from[current].position[0],current.position[1], came_from[current].position[1])
        current = came_from[current]
    path.append(start_station.name)
    path.reverse()
    return path,s*111

def dijkstra_search(start_station_name: str, end_station_name: str, map: dict[str, Station], distance_func=None) -> tuple[List[str], float]:
    if distance_func is None:
        distance_func = Euclidean_distance  # 默认使用欧几里得距离

    start_station = map[start_station_name]
    end_station = map[end_station_name]
    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    while not frontier.empty():
        current = frontier.get()[1]
        if current == end_station:
            break

        for next in current.links:
            new_cost = cost_so_far[current] + distance_func(current.position[0], next.position[0],
                                                            current.position[1], next.position[1])
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put((priority, next))
                came_from[next] = current

    path = []
    s = 0
    current = end_station
    while current != start_station:
        path.append(current.name)
        s += Euclidean_distance(current.position[0], came_from[current].position[0],
                                current.position[1], came_from[current].position[1])
        current = came_from[current]
    path.append(start_station.name)
    path.reverse()
    return path,s*111

def bfs_shortest_path(start_station_name: str, end_station_name: str, map: dict[str, Station], distance_func=None) -> tuple[List[str], float]:
    if distance_func is None:
        distance_func = Euclidean_distance

    start_station = map[start_station_name]
    end_station = map[end_station_name]
    frontier = Queue()
    frontier.put(start_station)
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    while not frontier.empty():
        current = frontier.get()
        if current == end_station:
            break

        for next in current.links:
            new_cost = cost_so_far[current] + distance_func(current.position[0], next.position[0],
                                                            current.position[1], next.position[1])
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                frontier.put(next)
                came_from[next] = current

    path = []
    s = 0
    current = end_station
    while current != start_station:
        path.append(current.name)
        s += Euclidean_distance(current.position[0], came_from[current].position[0],
                                current.position[1], came_from[current].position[1])
        current = came_from[current]
    path.append(start_station.name)
    path.reverse()
    return path,s*111


if __name__ == '__main__':

#     # 创建ArgumentParser对象
#     parser = argparse.ArgumentParser()
#     # 添加命令行参数
#     parser.add_argument('start_station_name', type=str, help='start_station_name')
#     parser.add_argument('end_station_name', type=str, help='end_station_name')
#     args = parser.parse_args()
#     start_station_name = args.start_station_name
#     end_station_name = args.end_station_name
    total_time1=0
    total_distance1=0
    total_time2=0
    total_distance2=0
    total_time3=0
    total_distance3=0
    total_time4=0
    total_distance4=0
    total_time5=0
    total_distance5=0
    for i in range(1):
        #random 任务时循环改成1000
        
        #start_station_name =  random.choice(column_data)
        #end_station_name = random.choice(column_data)
        #start_station_name =  "Amersham"
        #end_station_name = "Lewisham"
        start_station_name =  "Green Park"
        end_station_name = "Chigwell"
        #print(start_station_name,end_station_name)

        # The relevant descriptions of stations and underground_lines can be found in the build_data.py
        
        stations, underground_lines = build_data()
        start_time1 = time.time()
        path1,s1= get_path_As(start_station_name, end_station_name, stations)
        end_time1=time.time()
        total_time1=total_time1+end_time1-start_time1
        total_distance1=total_distance1+s1
        
        start_time2 = time.time()
        path2,s2= get_path_As(start_station_name, end_station_name, stations,Manhattan_distance)
        end_time2=time.time()
        total_time2=total_time2+end_time2-start_time2
        total_distance2=total_distance2+s2
        
        start_time3 = time.time()
        path3,s3= get_path_As(start_station_name, end_station_name, stations,Diagonal_distance)
        end_time3=time.time()
        total_time3=total_time3+end_time3-start_time3
        total_distance3=total_distance3+s3
        
        start_time4 = time.time()
        path4,s4= dijkstra_search(start_station_name, end_station_name, stations,Diagonal_distance)
        end_time4=time.time()
        total_time4=total_time4+end_time4-start_time4
        total_distance4=total_distance4+s4
        
        start_time5 = time.time()
        path5,s5= bfs_shortest_path(start_station_name, end_station_name, stations,Diagonal_distance)
        end_time5=time.time()
        total_time5=total_time5+end_time5-start_time5
        total_distance5=total_distance5+s5
        
        
        #print(f"Time:{1000*(end_time-start_time)}ms")
        # visualization the path
        # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
        plot_path(path1, f'visualization_underground/{start_station_name }{end_station_name}A*_Eu.html', stations, underground_lines)
        plot_path(path2, f'visualization_underground/{start_station_name }{end_station_name}A*_Mn.html', stations, underground_lines)
        plot_path(path3, f'visualization_underground/{start_station_name }{end_station_name}A*_Di.html', stations, underground_lines)
        plot_path(path4, f'visualization_underground/{start_station_name }{end_station_name}Dij.html', stations, underground_lines)
        plot_path(path5, f'visualization_underground/{start_station_name }{end_station_name}BFS.html', stations, underground_lines)
    print(total_time1*1000,total_distance1,total_time2*1000,total_distance2,total_time3*1000,total_distance3,total_time4*1000,total_distance4,total_time5*1000,total_distance5)