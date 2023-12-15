from typing import List
import heapq
from math import sqrt
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time


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
    
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    pass
    
    def dijkstra(start_station, end_station):
        # 初始化距离和前驱节点
        distance = {station: float('inf') for station in map.values()}
        distance[start_station] = 0
        prev = {}

        # 未访问的节点集合
        unvisited = set(map.values())

        while unvisited:
            current = min(unvisited, key=lambda station: distance[station])

            if current == end_station:
                path = reconstruct_path(prev, current)
                return path

            unvisited.remove(current)

            for neighbor in current.neighbors:
                if neighbor in unvisited:
                    # 计算新的距离
                    new_distance = distance[current] + 1
                    if new_distance < distance[neighbor]:
                        distance[neighbor] = new_distance
                        prev[neighbor] = current

        return None

    def astar(start_station, end_station, heuristic_fn):
        open_set = set()
        closed_set = set()
        g_score = {start_station: 0}
        f_score = {start_station: heuristic(start_station, end_station, heuristic_fn)}
        came_from = {}

        open_set.add(start_station)

        while open_set:
            current = min(open_set, key=lambda station: f_score[station])

            if current == end_station:
                path = reconstruct_path(came_from, current)
                return path

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in current.neighbors:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_station, heuristic_fn)

                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

    def heuristic(station1, station2, heuristic_fn):
        position1 = station1.position
        position2 = station2.position

        if heuristic_fn == 'manhattan':
            return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])
        elif heuristic_fn == 'euclidean':
            return ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
        else:
            raise ValueError("Invalid heuristic function. Choose either 'manhattan' or 'euclidean'")

    def reconstruct_path(came_from, current_station):
        path = [current_station.name]
        while current_station in came_from:
            current_station = came_from[current_station]
            path.insert(0, current_station.name)
        return path

    # 运行Dijkstra算法并测量执行时间
    start_time = time.time()
    dijkstra_path = dijkstra(start_station, end_station)
    dijkstra_time = time.time() - start_time

    # 运行A*算法(manhattan)并测量执行时间
    start_time = time.time()
    astar_path1 = astar(start_station, end_station, 'manhattan')
    astar_time1 = time.time() - start_time
    
    # 运行A*算法(euclidean)并测量执行时间
    start_time = time.time()
    astar_path2 = astar(start_station, end_station, 'euclidean')
    astar_time2 = time.time() - start_time

    # 比较路径长度和计算时间，选择最优的路径
    if dijkstra_path is not None and astar_path1 is not None and astar_path2 is not None:
        if len(dijkstra_path) < len(astar_path1) or (len(dijkstra_path) == len(astar_path1) 
                                                     and dijkstra_time < astar_time1):
            shortest_path = dijkstra_path
            shortest_time = dijkstra_time
            best_algorithm = "Dijkstra"
        elif len(astar_path1) < len(astar_path2) or (len(astar_path1) == len(astar_path2) 
                                                     and astar_time1 < astar_time2):
            shortest_path = astar_path1
            shortest_time = astar_time1
            best_algorithm = "A* (Manhattan)"
        else:
            shortest_path = astar_path2
            shortest_time = astar_time2
            best_algorithm = "A* (Euclidean)"
    elif dijkstra_path is not None and astar_path1 is not None:
        if len(dijkstra_path) < len(astar_path1) or (len(dijkstra_path) == len(astar_path1) 
                                                     and dijkstra_time < astar_time1):
            shortest_path = dijkstra_path
            shortest_time = dijkstra_time
            best_algorithm = "Dijkstra"
        else:
            shortest_path = astar_path1
            shortest_time = astar_time1
            best_algorithm = "A* (Manhattan)"
    elif dijkstra_path is not None and astar_path2 is not None:
        if len(dijkstra_path) < len(astar_path2) or (len(dijkstra_path) == len(astar_path2) 
                                                     and dijkstra_time < astar_time2):
            shortest_path = dijkstra_path
            shortest_time = dijkstra_time
            best_algorithm = "Dijkstra"
        else:
            shortest_path = astar_path2
            shortest_time = astar_time2
            best_algorithm = "A* (Euclidean)"
    elif dijkstra_path is not None:
        shortest_path = dijkstra_path
        shortest_time = dijkstra_time
        best_algorithm = "Dijkstra"
    else:
        shortest_path = astar_path1
        shortest_time = astar_time1
        best_algorithm = "A* (Manhattan)"

    print("algorithm:", "dijkstra")
    print("path:", dijkstra_path)
    print("Shortest path length:", len(dijkstra_path) - 1)
    print("Shortest path calculation time:", dijkstra_time)
    print("algorithm:", "A* (Manhattan)")
    print("path:", astar_path1)
    print("Shortest path length:", len(astar_path1) - 1)
    print("Shortest path calculation time:", astar_time1)
    print("algorithm:", "A* (Euclidean)")
    print("path:", astar_path2)
    print("Shortest path length:", len(astar_path2) - 1)
    print("Shortest path calculation time:", astar_time2)
    
    print("Best algorithm:", best_algorithm)
    print("Shortest path:", shortest_path)
    print("Shortest path length:", len(shortest_path) - 1)
    print("Shortest path calculation time:", shortest_time)

    return shortest_path
    
    


if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('--heuristic', type=str, choices=['manhattan', 'euclidean', 'dijkstra'], 
                        default='manhattan', help='heuristic function (default: manhattan)')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    heuristic_fn = args.heuristic

    
    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    filename = f"visualization_underground/my_shortest_path_{start_station_name}_{end_station_name}.html"
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    #plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    plot_path(path, filename , stations, underground_lines)
    