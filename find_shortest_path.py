from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse

from pympler import asizeof
import time
from algorithms import UCS,manhattan_Distance,Haversine_Distance,Astar,BFS,cal_total_cost



# Implement the following function
def get_path(start_station_name: str, end_station_name: str, algorithm:str, distance_func: str, map: dict[str, Station] ) -> List[str]:
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
    start_time = time.time()
    if algorithm == 'astar':
        path = Astar(start_station_name,end_station_name,map,distance_func)
    elif algorithm == 'ucs':
        path = UCS(start_station_name,end_station_name,map,distance_func)
    elif algorithm == 'bfs':
        path = BFS(start_station_name,end_station_name,map,distance_func)
    end_time = time.time()
    total_time = end_time - start_time
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    total_cost = cal_total_cost(path,map,distance_func)
    with open('output.txt','a') as f:
        f.write('Start station: '+start_station_name+'\n')
        f.write('End station: '+end_station_name+'\n')
        f.write('Algorithm: '+algorithm+'\n')
        f.write('Distance function: '+distance_func+'\n')
        f.write('Total cost: '+str(total_cost)+'\n')
        f.write('Path: '+'->'.join(path)+'\n')
        f.write('Time: '+str(total_time)+'\n')
        f.write('\n')


    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    return path


if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('algorithm', type=str, default='astar', help='algorithm')
    parser.add_argument('distance_func', type=str, default='manhattan', help='distance_func')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    algorithm = args.algorithm
    distance_func = args.distance_func

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, algorithm, distance_func, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
