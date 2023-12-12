import math
import timeit
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue


def Euclidean_distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[
    str]:
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
            new_cost = cost_so_far[current] + Euclidean_distance(current.position[0], next.position[0],
                                                                      current.position[1], next.position[1])
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
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
    print(f'搜索的格子数为 {len(came_from)}')
    print(f'路径的长度为{s * 111: .3f} km')
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

    start_time = timeit.default_timer()

    path = dijkstra(start_station_name, end_station_name, stations)
    end_time = timeit.default_timer()

    # 计算执行时间
    print(f"Dijkstra 执行时间：{(end_time - start_time) * 1000:.3f} ms")

    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/Dijkstra_in_London_railway.html', stations,
              underground_lines)
