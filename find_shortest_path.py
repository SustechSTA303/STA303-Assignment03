from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from Astar import Astar
from Dijkstra import Dijkstra
from Bidirectional import Bidirectional
import time
import math
from geopy.distance import geodesic

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
    #start_station = map[start_station_name]
    #end_station = map[end_station_name]
    # start_time = time.time()
    # path_length = 0
    # for key_1 in map:
    #     for key_2 in map:
    #         if key_1 == key_2:
    #             continue    
    #         path_list = Astar(key_1, key_2, map)
    #         path_length += calLength('euclid', path_list, map)
    # end_time = time.time()
    # run_time = end_time - start_time

    start_time = time.time()
    path_list = Astar(start_station_name, end_station_name, map)
    end_time = time.time()
    run_time = end_time - start_time

    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    #print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    #print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    print(f"The path length is {calLength('euclid', path_list, map)}")
    print(f"Algorithm took {run_time} seconds to run.")
    return path_list

def calLength(type, path_list, map):
    count = 0
    length = 0
    while len(path_list) > count + 1:
        node_1 = map[path_list[count]]
        node_2 = map[path_list[count+1]]
        node_dis = distance(node_1, node_2, type)
        length += node_dis
        count += 1
    return length

def distance(node_1, node_2, type):
    node_1_location = (node_1.position[0], node_1.position[1])
    node_2_location = (node_2.position[0], node_2.position[1])
    if type == 'manhattan':
        return abs(node_1_location[0] - node_2_location[0]) + abs(node_1_location[1] + node_2_location[1])
    elif type == 'euclid':
        return math.sqrt((node_1_location[0] - node_2_location[0])**2 + (node_1_location[1] + node_2_location[1])**2)
    elif type == 'geo':
        return geodesic(node_1_location, node_2_location).kilometers




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
    path = get_path(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
