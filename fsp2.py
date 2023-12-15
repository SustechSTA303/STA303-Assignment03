import time
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import sys


def initialize_single_source(graph, start):
    """
    初始化图的单源最短路径估计和前驱节点
    """
    distance = {node: sys.maxsize for node in graph}
    predecessor = {node: None for node in graph}
    distance[start] = 0
    return distance, predecessor


def relax(u, v, weight, distance, predecessor):
    """
    松弛操作
    """
    if distance[v] > distance[u] + weight:
        distance[v] = distance[u] + weight
        predecessor[v] = u


def bellman_ford(graph, start):
    """
    使用Bellman-Ford算法找到图中从起点到其他节点的最短路径
    """
    distance, predecessor = initialize_single_source(graph, start)

    # 记录开始时间
    start_time = time.time()

    # 进行|V| - 1次松弛操作
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u].items():
                relax(u, v, weight, distance, predecessor)

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Bellman-Ford algorithm execution time              : {elapsed_time} seconds")

    # 检查是否存在负权环
    for u in graph:
        for v, weight in graph[u].items():
            if distance[v] > distance[u] + weight:
                raise ValueError("Graph contains a negative-weight cycle")

    return distance, predecessor


def get_shortest_path(predecessor, start, end):
    """
    根据前驱节点字典重建最短路径
    """
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = predecessor[current]
    return path


def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    # 构建图的邻接字典
    graph = {station.name: {neighbor.name: 1 for neighbor in station.links} for station in map.values()}

    start_station = start_station_name
    end_station = end_station_name

    # 使用Bellman-Ford算法计算最短路径
    distance, predecessor = bellman_ford(graph, start_station)

    # 重建最短路径
    shortest_path = get_shortest_path(predecessor, start_station, end_station)

    return shortest_path


if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    # 有关站点和地铁线的相关描述可以在build_data.py中找到
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    # 可视化路径
    # 打开visualization_underground/my_path_in_London_railway.html查看路径，你的路径将以红色标记
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
