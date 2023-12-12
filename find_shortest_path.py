import math
import timeit
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue


def Euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def Manhattan_distance(x1, x2, y1, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def Diagonal_distance(x1, x2, y1, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)


def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], distance_func=None) -> List[
    str]:
    """
    runs astar on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): str The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
        distance_func(function): The heuristic function used in A*
    Returns:
        List[Station]: A path composed of a series of station_name
    """

    if distance_func is None:
        distance_func = Euclidean_distance  # 默认使用欧几里得距离

    start_station = map[start_station_name]
    end_station = map[end_station_name]
    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    while not frontier.empty():
        current = frontier.get()[1]
        if current == end_station:
            break

        for next in current.links:
            new_cost = cost_so_far[current] + distance_func(current.position[0], next.position[0],
                                                            current.position[1], next.position[1])
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + distance_func(next.position[0], end_station.position[0],
                                                    next.position[1], end_station.position[1])
                frontier.put((priority, next))
                came_from[next] = current

    path = []
    s = 0
    current = end_station
    while current != start_station:
        path.append(current.name)
        s += Euclidean_distance(current.position[0], came_from[current].position[0],
                                current.position[1], came_from[current].position[1])
        current = came_from[current]
    path.append(start_station.name)
    path.reverse()
    print(f'{distance_func.__name__}搜索的格子数为 {len(came_from)}')
    print(f'{distance_func.__name__}路径的长度为{s * 111: .3f} km')
    return path


if __name__ == '__main__':
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    stations, underground_lines = build_data()

    start_time_e = timeit.default_timer()
    path_e = get_path(start_station_name, end_station_name, stations, Euclidean_distance)
    end_time_e = timeit.default_timer()
    start_time_m = timeit.default_timer()
    path_m = get_path(start_station_name, end_station_name, stations, Manhattan_distance)
    end_time_m = timeit.default_timer()
    start_time_d = timeit.default_timer()
    path_d = get_path(start_station_name, end_station_name, stations, Diagonal_distance)
    end_time_d = timeit.default_timer()

    # 计算执行时间
    print(f"Euclidean_distance 执行时间：{(end_time_e - start_time_e) * 1000:.3f} ms")
    print(f"Manhattan_distance 执行时间：{(end_time_m - start_time_m) * 1000:.3f} ms")
    print(f"Diagonal_distance 执行时间：{(end_time_d - start_time_d) * 1000:.3f} ms")

    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path_e, 'visualization_underground/my_shortest_path_e_in_London_railway.html', stations,
              underground_lines)
    plot_path(path_m, 'visualization_underground/my_shortest_path_m_in_London_railway.html', stations,
              underground_lines)
    plot_path(path_d, 'visualization_underground/my_shortest_path_d_in_London_railway.html', stations,
              underground_lines)
