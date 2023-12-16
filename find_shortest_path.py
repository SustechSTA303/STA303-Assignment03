from typing import List, Dict, Set
from plot_underground_path import plot_path
from build_data import build_data, Station
from math import inf
import argparse
import heapq
import time


def get_path(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs A* on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[str]: A path composed of a series of station names
    """
#欧几里得
    def heuristic(station_a: Station, station_b: Station) -> float:
        # Define a heuristic function here (e.g., Euclidean distance between positions)
        # For example:
        return ((station_a.position[0] - station_b.position[0]) ** 2 +
                (station_a.position[1] - station_b.position[1]) ** 2) ** 0.5


    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = []
    closed_set = set()
    came_from = {}

    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    f_score = {station: float('inf') for station in map.values()}
    f_score[start_station] = heuristic(start_station, end_station)

    heapq.heappush(open_set, (f_score[start_station], start_station))

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end_station:
            path = []
            while current in came_from:
                path.append(current.name)
                current = came_from[current]
            path.append(start_station.name)
            return path[::-1]

        closed_set.add(current)

        for neighbor in current.links:  # Assuming 'links' contains neighboring stations
            tentative_g_score = g_score[current] + 1  # Assuming each link has a weight of 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_station)
                if neighbor not in closed_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()
    
    # 计算并输出 A* 算法的运行时间
    start_time = time.time()
    path = get_path(start_station_name, end_station_name, stations)
    print(path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"A* Algorithm using Euclidean Execution Time: {execution_time} seconds")

    path = get_path(start_station_name, end_station_name, stations)
    plot_path(path, 'visualization_underground/my_shortest_pathA*_in_London_railway_usnig_Euclidean.html', stations, underground_lines)


######################################################################################


def get_path_a_star(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs A* on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[str]: A path composed of a series of station names
    """
    # 更新启发式函数为曼哈顿距离
    def heuristic(station_a: Station, station_b: Station) -> float:
        return abs(station_a.position[0] - station_b.position[0]) + abs(station_a.position[1] - station_b.position[1])



    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = []
    closed_set = set()
    came_from = {}

    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    f_score = {station: float('inf') for station in map.values()}
    f_score[start_station] = heuristic(start_station, end_station)



    heapq.heappush(open_set, (f_score[start_station], start_station))

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end_station:
            path = []
            while current in came_from:
                path.append(current.name)
                current = came_from[current]
            path.append(start_station.name)
            return path[::-1]

        closed_set.add(current)

        for neighbor in current.links:  # Assuming 'links' contains neighboring stations
            tentative_g_score = g_score[current] + 1  # Assuming each link has a weight of 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_station)
                if neighbor not in closed_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()
    
    # 计算并输出 A* 算法的运行时间
    start_time = time.time()
    path = get_path_a_star(start_station_name, end_station_name, stations)
    print(path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"A* Algorithm using manhattan Execution Time: {execution_time} seconds")
    path = get_path(start_station_name, end_station_name, stations)
    plot_path(path, 'visualization_underground/my_shortest_path_A*_in_London_railway_using_manhattan.html', stations, underground_lines)
####################################################################################

def get_path(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs A* on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
    Returns:
        List[str]: A path composed of a series of station names
    """
    def chebyshev_heuristic(station_a: Station, station_b: Station) -> float:
    # 计算切比雪夫距离
        return max(abs(station_a.position[0] - station_b.position[0]), abs(station_a.position[1] - station_b.position[1]))



    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = []
    closed_set = set()
    came_from = {}

    g_score = {station: float('inf') for station in map.values()}
    g_score[start_station] = 0
    f_score = {station: float('inf') for station in map.values()}
    f_score[start_station] = chebyshev_heuristic(start_station, end_station)

    heapq.heappush(open_set, (f_score[start_station], start_station))

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end_station:
            path = []
            while current in came_from:
                path.append(current.name)
                current = came_from[current]
            path.append(start_station.name)
            return path[::-1]

        closed_set.add(current)

        for neighbor in current.links:  # Assuming 'links' contains neighboring stations
            tentative_g_score = g_score[current] + 1  # Assuming each link has a weight of 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + chebyshev_heuristic(neighbor, end_station)
                if neighbor not in closed_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()
    
    # 计算并输出 A* 算法的运行时间
    start_time = time.time()
    path = get_path(start_station_name, end_station_name, stations)
    print(path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"A* Algorithm using chebyshev Execution Time: {execution_time} seconds")

    path = get_path(start_station_name, end_station_name, stations)
    plot_path(path, 'visualization_underground/my_shortest_pathA*_in_London_railway_usnig_chebyshev.html', stations, underground_lines)
    
    
    
    
def bellman_ford(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    distances = {station: inf for station in map.values()}
    distances[start_station] = 0
    prev = {station: None for station in map.values()}

    # Relax edges repeatedly
    for _ in range(len(map) - 1):
        for station in map.values():
            for neighbor in station.links:
                if distances[station] + 1 < distances[neighbor]:  # Assuming each link has a weight of 1
                    distances[neighbor] = distances[station] + 1
                    prev[neighbor] = station

    # Check for negative cycles
    for station in map.values():
        for neighbor in station.links:
            if distances[station] + 1 < distances[neighbor]:  # Assuming each link has a weight of 1
                raise ValueError("Graph contains negative cycle")

    # Retrieve shortest path
    path = []
    current = end_station
    while current:
        path.append(current.name)
        current = prev[current]

    return path[::-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()

 # 计算并输出算法的运行时间
    start_time = time.time()
    path = bellman_ford(start_station_name, end_station_name, stations)
    end_time = time.time()
    execution_time = end_time - start_time
    path = bellman_ford(start_station_name, end_station_name, stations)
    print(path)  # 输出找到的路径
    print(f"bellman_ford Algorithm Execution Time: {execution_time} seconds")
    plot_path(path, 'visualization_underground/my_shortest_path_using_bellman_ford.html', stations, underground_lines)

    
    
    
###########################################################
def dijkstra_shortest_path(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs Dijkstra's algorithm on the map to find the shortest path between start and end stations
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name
    Returns:
        List[str]: A path composed of a series of station names
    """
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Initialize data structures
    distances = {station: sys.maxsize for station in map.values()}
    distances[start_station] = 0
    pq = [(0, start_station)]
    visited = set()

    while pq:
        current_distance, current_station = heapq.heappop(pq)

        if current_station == end_station:
            path = []
            while current_station is not None:
                path.append(current_station.name)
                current_station = current_station.prev
            return path[::-1]

        if current_station in visited:
            continue

        visited.add(current_station)

        for neighbor in current_station.links:  # Assuming 'links' contains neighboring stations
            distance_to_neighbor = current_distance + 1  # Assuming each link has a weight of 1
            if distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = distance_to_neighbor
                neighbor.prev = current_station
                heapq.heappush(pq, (distance_to_neighbor, neighbor))

    return []
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()

 # 计算并输出算法的运行时间
    start_time = time.time()
    path = bellman_ford(start_station_name, end_station_name, stations)
    end_time = time.time()
    execution_time = end_time - start_time
    path = bellman_ford(start_station_name, end_station_name, stations)
    print(path)  # 输出找到的路径
    print(f"dijkstra Algorithm Execution Time: {execution_time} seconds")
    plot_path(path, 'visualization_underground/my_shortest_path_using_dijkstra.html', stations, underground_lines)