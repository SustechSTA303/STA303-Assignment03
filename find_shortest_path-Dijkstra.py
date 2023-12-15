
import math
from queue import PriorityQueue, Queue
import timeit
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse

def L2_distance(station1: Station, station2: Station) -> float:
    """Calculate the L2 Norm
    """
    return math.sqrt((station1.position[0] - station2.position[0]) ** 2 + (station1.position[1] - station2.position[1]) ** 2)

#Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    Runs A* on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): str The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    
    frontier = PriorityQueue()
    frontier.put((0,start_station))
    
    came_from = {}
    came_from[start_station] = None
    
    while not frontier.empty():
        _, current = frontier.get()
        
        if current == end_station:
            break
        
        for next_station in map[current.name].links:
            if next_station not in came_from:
                 #L2距离
                priority = L2_distance(end_station,next_station)


                frontier.put((priority,next_station))
                came_from[next_station] = current

    my_path = []
    current = end_station
    while current is not None:
        my_path.append(current.name)
        current = came_from[current]
    my_path.reverse()
    
    return my_path

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
    path = []
    # 计时
    start_time = timeit.default_timer()
    path = get_path(start_station_name, end_station_name, stations)
    end_time = timeit.default_timer()
    print(end_time - start_time)
    print(path)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)


