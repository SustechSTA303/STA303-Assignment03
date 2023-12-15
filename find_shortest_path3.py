import time
from typing import List
import heapq
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse


class Station:
    """
    站点类包含四个属性：id、name、position、links和came_from。
    Position是经度和纬度的二进制组合
    Links是与站点对象相邻的站点列表
    """
    def __init__(self, id, name, position):
        self.id = id
        self.name = name
        self.position = position
        self.links = set()
        self.came_from = None


def heuristic_cost_estimate(current, goal):
    """
    A heuristic function to estimate the cost from the current station to the goal station.
    You can customize this function based on your requirements.
    """
    # Example: Euclidean distance between two stations' positions
    return ((current.position[0] - goal.position[0]) ** 2 +
            (current.position[1] - goal.position[1]) ** 2) ** 0.5

def heuristic_cost_estimate_manhattan(current, goal):
    """
    A heuristic function using Manhattan distance to estimate the cost from the current station to the goal station.
    """
    return abs(current.position[0] - goal.position[0]) + abs(current.position[1] - goal.position[1])

def heuristic_cost_estimate_chebyshev(current, goal):
    """
    A heuristic function using Chebyshev distance to estimate the cost from the current station to the goal station.
    """
    return max(abs(current.position[0] - goal.position[0]), abs(current.position[1] - goal.position[1]))


def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_time = time.time()

    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = [(0, id(start_station),start_station)]
    g_score = {start_station: 0}
    f_score = {start_station: heuristic_cost_estimate_chebyshev(start_station, end_station)}

    while open_set:
        current_cost,_, current_station = heapq.heappop(open_set)

        if current_station == end_station:
            path = [current_station.name]
            while current_station != start_station:
                current_station = current_station.came_from
                path.insert(0, current_station.name)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"A* with Chebyshev distance algorithm execution time: {elapsed_time} seconds")
            return path

        for neighbor in current_station.links:
            tentative_g_score = g_score[current_station] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_cost_estimate_chebyshev(neighbor, end_station)
                heapq.heappush(open_set, (f_score[neighbor],id(neighbor), neighbor))
                neighbor.came_from = current_station

    return []


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
