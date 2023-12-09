from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from algs import astar
from algs import greedy_bfs
from algs import dijkstra


# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], algorithm: str, heuristic: str) -> (List[str], float):
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
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    # print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    # print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    ################### The following is the code student implemented ###################
    # path = astar(start_station, end_station, map)
    # path = greedy_bfs(start_station, end_station, map)
    if algorithm == 'astar':
        path, path_distance, clo_size = astar(start_station, end_station, heuristic)
    elif algorithm == 'greedy_bfs':
        path, path_distance, clo_size = greedy_bfs(start_station, end_station, heuristic)
    elif algorithm == 'dijikstra':
        path, path_distance, clo_size = dijkstra(start_station, end_station)
    else:
        raise Exception("Invalid algorithm")
    return path, path_distance, clo_size
    


if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('algorithm', type=str, default='astar', help='algorithm')
    parser.add_argument('heuristic', type=str, default='euclidean', help='heuristic')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    algorithm = args.algorithm
    heuristic = args.heuristic

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path, path_distance, clo_size = get_path(start_station_name, end_station_name, stations, algorithm, heuristic)
    print(f'The closet size is {clo_size}')
    print(f'The distance of the shortest path between {start_station_name} and {end_station_name} is {path_distance} km')
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
