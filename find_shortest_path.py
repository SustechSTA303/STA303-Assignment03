from typing import List,Dict
from plot_underground_path import plot_path
from build_data import Station, build_data, build_adjacency_matrix
import argparse, math
from heapq import heappop, heappush
from sys import maxsize

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    在地图上运行A*算法，找到起始站点和终点站点之间的最短路径
    参数:
        start_station_name(str): 起始站点的名称
        end_station_name(str): 终点站点的名称
        map(dict[str, Station]): 站点名称与站点对象的映射，
                                请参考build_data.py中关于Station类的相关注释描述
    返回:
        List[Station]: 由一系列站点名称组成的路径
    """
    # 从地图中获取起始站和终点站
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # 字典，用于存储从起点到每个站点的实际代价
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0

    # 字典，用于存储从每个站点到终点的启发式代价
    f_scores = {station: float('inf') for station in map.values()}
    f_scores[start_station] = chebyshev_distance(start_station, end_station)

    # 字典，用于存储路径
    came_from = {}

    # 优先队列，用于按照估计代价排序
    open_set = [(f_scores[start_station], id(start_station), start_station)]

    while open_set:
        current_f_score, _, current_station = heappop(open_set)

        if current_station == end_station:
            # 重建路径
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            return path

        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 1
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current_station
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + chebyshev_distance(neighbor, end_station)

                # 如果相邻站点不在优先队列中，将其加入
                if neighbor not in [station for _, _, station in open_set]:
                    heappush(open_set, (f_scores[neighbor], id(neighbor), neighbor))

    return []


def dijkstra(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    运行Dijkstra算法，找到起始站点和终点站点之间的最短路径。
    参数:
        start_station_name(str): 起始站点的名称。
        end_station_name(str): 终点站点的名称。
        map(Dict[str, Station]): 站点名称与站点对象的映射。
    返回:
        List[str]: 由一系列站点名称组成的路径。
    """
    # 从地图中获取起始和终点站点
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # 字典，用于存储从起点到每个站点的实际代价
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0

    # 优先队列，根据当前代价对站点进行排序
    open_set = [(0, id(start_station), start_station)]

    # 字典，用于存储最优路径中前一个站点
    came_from = {}

    while open_set:
        current_g_score, _, current_station = heappop(open_set)

        if current_station == end_station:
            # 重构路径
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            return path

        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 0  # 假设所有边的代价相等
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current_station
                g_scores[neighbor] = tentative_g_score

                # 如果邻居不在优先队列中，将其添加进去
                heappush(open_set, (g_scores[neighbor], id(neighbor), neighbor))

    return []

def bellman_ford(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    运行 Bellman-Ford 算法在地图上，找到 start_station_name 和 end_station_name 之间的最短路径。
    Args:
        start_station_name(str): 起始站点的名称。
        end_station_name(str): 结束站点的名称。
        map(Dict[str, Station]): 站点名称与对应的站点对象的映射。
    Returns:
        List[str]: 由一系列站点名称组成的路径。
    """
    # 从地图中获取起始站和终点站
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # 用于存储从起点到每个站点的实际代价的字典
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0
    
    # 字典，用于存储路径
    came_from = {}

    # 通过松弛操作重复地进行边的松弛
    for _ in range(len(map) - 1):
        for current_station in map.values():
            for neighbor_name in current_station.links:
                neighbor = map[neighbor_name.name]
                distance = 1  # 假设所有边的权重相等
                tentative_g_score = g_scores[current_station] + distance

                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    came_from[neighbor] = current_station

    # 检查是否存在负权重环
    for current_station in map.values():
        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 1  # 假设所有边的权重相等
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                raise ValueError("图中存在负权重环")

    # 重建路径
    path = []
    current_station = end_station
    while current_station:
        path.insert(0, current_station.name)
        current_station = came_from.get(current_station)

    return path

def spfa(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    运行 SPFA 算法在地图上，找到 start_station_name 和 end_station_name 之间的最短路径。
    Args:
        start_station_name(str): 起始站点的名称。
        end_station_name(str): 结束站点的名称。
        map(Dict[str, Station]): 站点名称与对应的站点对象的映射。
    Returns:
        List[str]: 由一系列站点名称组成的路径。
    """
    # 从地图中获取起始站和终点站
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # 用于存储从起点到每个站点的实际代价的字典
    g_scores = {station: float('inf') for station in map.values()}
    g_scores[start_station] = 0
    
    # 字典，用于存储路径
    came_from = {}

    # 优化队列，用于按照估计代价排序
    open_set = [(0,id(start_station), start_station)]

    while open_set:
        current_g_score,_, current_station = heappop(open_set)
        
        if current_station == end_station:
            # 重建路径
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from.get(current_station)
            return path

        for neighbor_name in current_station.links:
            neighbor = map[neighbor_name.name]
            distance = 1  # 假设所有边的权重相等
            tentative_g_score = g_scores[current_station] + distance

            if tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current_station
                g_scores[neighbor] = tentative_g_score

                # 如果相邻站点不在优化队列中，将其加入
                if neighbor not in [station for _,_, station in open_set]:
                    heappush(open_set, (g_scores[neighbor],id(neighbor), neighbor))

    return []

def floyd_warshall(start_station_name: str, end_station_name: str, map: Dict[str, Station]) -> List[str]:
    """
    运行Floyd-Warshall算法，找到起始站点和终点站点之间的最短路径。
    参数:
        start_station_name(str): 起始站点的名称。
        end_station_name(str): 终点站点的名称。
        map(Dict[str, Station]): 站点名称与站点对象的映射。
    返回:
        List[str]: 由一系列站点名称组成的路径。
    """
    names, adjacency_matrix = build_adjacency_matrix(map)
    
    # 获取起始和终点站点的索引
    start_index = names.index(start_station_name)
    end_index = names.index(end_station_name)

    num_stations = len(names)

    # 初始化用于重构路径的next矩阵
    next_matrix = [[None] * num_stations for _ in range(num_stations)]
    for i in range(num_stations):
        for j in range(num_stations):
            next_matrix[i][j] = j if adjacency_matrix[i][j] != float('inf') else None

    # Floyd-Warshall算法
    for k in range(num_stations):
        for i in range(num_stations):
            for j in range(num_stations):
                if adjacency_matrix[i][k] + adjacency_matrix[k][j] < adjacency_matrix[i][j]:
                    adjacency_matrix[i][j] = adjacency_matrix[i][k] + adjacency_matrix[k][j]
                    next_matrix[i][j] = next_matrix[i][k]

    # 重构路径
    path = []
    current_index = start_index
    while current_index != end_index:
        path.append(names[current_index])
        current_index = next_matrix[current_index][end_index]

    path.append(names[end_index])
    return path

# 启发式函数，例如两站点之间的直线距离
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
    path = spfa(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)