from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq 
import algorithm  #the algorithms are in this file.
import queue
import time



Cost=['UNIT','DISTANCE'] #UNIT represents that all cost between two adjacent stations is 1. DISTANCE represents using the realistic distance as the cost.
algorithms=dict()
algorithms[algorithm.get_path_A_Manhatton]="A*_Manhatton"
algorithms[algorithm.get_path_A_L2]="A*_Euclidean"
algorithms[algorithm.get_path_dijkstra]="Dijkstra"
algorithms[algorithm.get_path_bfs]="BFS"   


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
    with open("output.txt","a") as f:
        f.write(f"the start station is {start_station_name},the end station is {end_station_name}.\n")
        
    for cost in Cost:
        for key in algorithms:
            time_start=time.time()
            path=key(start_station_name, end_station_name, stations,cost)
            time_end=time.time()
            print(f"the cost time is {(time_end-time_start)*1000} ms, the cost is calculated through {cost}, the algorithm used is {algorithms[key]}. The length of the path is {len(path)}. ")
            with open("output.txt","a") as f:
                f.write(f"the cost time is {(time_end-time_start)*1000} ms, the cost is calculated through {cost}, the algorithm used is {algorithms[key]}.The length of the path is {len(path)}.\n ")
            
        
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
