from typing import List,Dict
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse, math
from heapq import heappop, heappush


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
    # 从地图中获取起始站和终点站
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # 字典，用于存储从起点到每个站点的实际代价
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0

    # 字典，用于存储从每个站点到终点的启发式代价
    f_scores = {station: float('inf') for station in map.values()}
    f_scores[start_station] = heuristic(start_station, end_station)

    # 字典，用于存储路径
    came_from = {}

    # 优先队列，用于按照估计代价排序
    open_set = [(f_scores[start_station], start_station)]

    while open_set:
        current_f_score, current_station = heapq.heappop(open_set)
        
        if current_station == end_station:
            # 重建路径
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            return path

        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 0
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current_station
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic(neighbor, end_station)

                # 如果相邻站点不在优先队列中，将其加入
                if neighbor not in [station for _, station in open_set]:
                    heapq.heappush(open_set, (f_scores[neighbor], neighbor))

    return []

def dijkstra(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs Dijkstra's algorithm on the map, finds the shortest path between start_station_name and end_station_name.
    Args:
        start_station_name(str): The name of the starting station.
        end_station_name(str): The name of the ending station.
        map(Dict[str, Station]): Mapping between station names and station objects of the name.
    Returns:
        List[str]: A path composed of a series of station names.
    """
    # Get the starting and ending stations from the map
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Dictionary to store the actual cost from the start to each station
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0

    # Priority queue to sort stations based on their current cost
    open_set = [(0, id(start_station), start_station)]

    # Dictionary to store the previous station in the optimal path
    came_from = {}

    while open_set:
        current_g_score, _, current_station = heappop(open_set)

        if current_station == end_station:
            # Reconstruct the path
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            return path

        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 0  # Assuming equal cost for all edges
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current_station
                g_scores[neighbor] = tentative_g_score

                # If neighbor is not in the priority queue, add it
                heappush(open_set, (g_scores[neighbor], id(neighbor), neighbor))

    return []

def uniform_cost_search(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    Runs Uniform Cost Search (UCS) algorithm on the map, finds the shortest path between start_station_name and end_station_name.
    Args:
        start_station_name(str): The name of the starting station.
        end_station_name(str): The name of the ending station.
        map(Dict[str, Station]): Mapping between station names and station objects of the name.
    Returns:
        List[str]: A path composed of a series of station names.
    """
    # Get the starting and ending stations from the map
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Dictionary to store the actual cost from the start to each station
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0

    # Priority queue to sort stations based on their current cost
    open_set = [(0, id(start_station), start_station)]

    # Dictionary to store the previous station in the optimal path
    came_from = {}

    while open_set:
        current_g_score, _, current_station = heappop(open_set)

        if current_station == end_station:
            # Reconstruct the path
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            return path

        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 0  # Assuming equal cost for all edges
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current_station
                g_scores[neighbor] = tentative_g_score

                # If neighbor is not in the priority queue, add it
                heappush(open_set, (g_scores[neighbor], id(neighbor), neighbor))

    return []

# 启发式函数，例如两站点之间的直线距离
def heuristic(station1: Station, station2: Station) -> float:
    return manhattan_distance(station1, station2)

def euclidean_distance(station1: Station, station2: Station) -> float:
    x1, y1 = station1.position
    x2, y2 = station2.position
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def manhattan_distance(station1: Station, station2: Station) -> float:
    x1, y1 = station1.position
    x2, y2 = station2.position
    return abs(x2 - x1) + abs(y2 - y1)

def chebyshev_distance(station1: Station, station2: Station) -> float:
    x1, y1 = station1.position
    x2, y2 = station2.position
    return max(abs(x2 - x1), abs(y2 - y1))

def haversine_distance(station1: Station, station2: Station) -> float:
    # Assuming station.position is (latitude, longitude)
    lat1, lon1 = station1.position
    lat2, lon2 = station2.position
    # Haversine formula
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

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
    path = uniform_cost_search(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)