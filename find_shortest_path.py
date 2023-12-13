from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue
import random


# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict) -> List[str]:
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
    # # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    # print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    # print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    # pass

    frontier = PriorityQueue ()
    frontier.put((0,start_station.name)) 
    came_from = {}
    cost_so_far = {}
    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0
    while not frontier.empty() :
        current = map[frontier.get()[1]]
        if current == end_station:
            break
        for next in stations[current.name].links :
            new_cost = cost_so_far[current.name] + distance(stations[current.name].position, stations[next.name].position)
            if next.name not in cost_so_far or new_cost < cost_so_far[next.name]:
                cost_so_far[next.name] = new_cost
                priority = new_cost + distance(end_station.position, next.position)
                # priority = distance(end_station.position, next.position)
                frontier.put((priority, next.name))
                came_from[next.name] = current

    tmpsta = end_station
    path = []
    while type(tmpsta) is not type(None):
        path.append(tmpsta.name)
        tmpsta = came_from[tmpsta.name]
    return path, len(cost_so_far)

def distance(a,b):
    dis = float(abs(a[0]-b[0])+abs(a[1]-b[1]))
    return dis

def get_path_DFS(start_station_name: str, end_station_name: str, map: dict) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue ()
    frontier.put((0,start_station.name)) 
    came_from = {}
    cost_so_far = {}
    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0
    while not frontier.empty() :
        current = map[frontier.get()[1]]
        if current == end_station:
            break
        for next in stations[current.name].links :
            new_cost = cost_so_far[current.name] + distance(stations[current.name].position, stations[next.name].position)
            if next.name not in cost_so_far or new_cost < cost_so_far[next.name]:
                cost_so_far[next.name] = new_cost
                # priority = new_cost + distance(end_station.position, next.position)
                priority = distance(end_station.position, next.position)
                frontier.put((priority, next.name))
                came_from[next.name] = current

    tmpsta = end_station
    path = []

    while type(tmpsta) is not type(None):
        path.append(tmpsta.name)
        tmpsta = came_from[tmpsta.name]
    print(path)
    return path, len(cost_so_far)


def get_path_BFS(start_station_name: str, end_station_name: str, map: dict) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    frontier = PriorityQueue ()
    frontier.put((0,start_station.name)) 
    came_from = {}
    cost_so_far = {}
    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0
    while not frontier.empty() :
        current = map[frontier.get()[1]]
        if current == end_station:
            break
        for next in stations[current.name].links :
            new_cost = cost_so_far[current.name] + distance(stations[current.name].position, stations[next.name].position)
            if next.name not in cost_so_far or new_cost < cost_so_far[next.name]:
                cost_so_far[next.name] = new_cost
                priority = new_cost
                frontier.put((priority, next.name))
                came_from[next.name] = current

    tmpsta = end_station
    path = []

    while type(tmpsta) is not type(None):
        path.append(tmpsta.name)
        tmpsta = came_from[tmpsta.name]
    print(path)
    return path, len(cost_so_far)


if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    print(start_station_name, end_station_name)

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()

    # For random selection experiment
    # num_search_ave=0
    # path_length=0
    # BFS_more_search=0
    # BFS_path_len=0
    # DFS_more_search=0
    # DFS_path_len=0
    # for i in range(100):
    #     print(i)
    #     random_key1 = random.choice(list(stations.keys()))
    #     random_key2 = random.choice(list(stations.keys()))
    #     print(random_key1,"to",random_key2)
    #     path, num_search= get_path(random_key1, random_key2, stations)
    #     path_BFS, num_search_BFS= get_path_BFS(random_key1, random_key2, stations)
    #     path_DFS, num_search_DFS= get_path_DFS(random_key1, random_key2, stations)

    #     BFS_more_search = BFS_more_search + num_search_BFS - num_search
    #     DFS_more_search = DFS_more_search + num_search_DFS - num_search
    #     num_search_ave = num_search_ave+num_search
    #     path_length = path_length+len(path)
    #     BFS_path_len = BFS_path_len+len(path_BFS)
    #     DFS_path_len = DFS_path_len+len(path_DFS)

    # print("average search number",num_search_ave/100)
    # print("average more search of BFS",BFS_more_search/100)
    # print("average more search of DFS",DFS_more_search/100)
    # print("average A* path length", path_length/100)
    # print("average BFS path length", BFS_path_len/100)
    # print("average DFS path length", DFS_path_len/100)


    path, _ = get_path(start_station_name, end_station_name, stations)
    print(path)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
