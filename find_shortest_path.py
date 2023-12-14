import heapq
from queue import PriorityQueue
from typing import List, Any
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from collections import deque


def abs_heuristic(next_station: Station, end_station: Station):
    n1 = float(next_station.position[0])
    n2 = float(next_station.position[1])
    e1 = float(end_station.position[0])
    e2 = float(end_station.position[1])
    distance = ((n1 - n2) ** 2 + (e1 - e2) ** 2) ** 0.5

    return distance


def M_heuristic(next_station: Station, end_station: Station):
    n1 = float(next_station.position[0])
    n2 = float(next_station.position[1])
    e1 = float(end_station.position[0])
    e2 = float(end_station.position[1])
    distance = abs(n1 - n2) + abs(e1 - e2)

    return distance


# Implement the following function
def heuristic(next_station: Station, end_station: Station, method='M'):
    if method == 'M': return M_heuristic(next_station, end_station)
    return abs_heuristic(next_station, end_station)


def astar_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], method='M') -> List[str]:
    end_station = map[end_station_name]

    open_set = PriorityQueue()
    open_set.put((0, start_station_name))
    came_from = {start_station_name: None}
    g_score = {station: float('inf') for station in map}
    g_score[start_station_name] = 0

    while not open_set.empty():
        _, current_name = open_set.get()
        current_station = map[current_name]

        if current_name == end_station_name:
            path = []
            while current_name:
                path.append(current_name)
                current_name = came_from[current_name]
            return path[::-1]

        for neighbor in current_station.links:
            neighbor_name = neighbor.name
            tentative_g_score = g_score[current_name] + heuristic(neighbor, current_station, method)

            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, end_station, method)
                open_set.put((f_score, neighbor_name))

    return []


def bfs_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> list[Any]:
    open_station = deque([(start_station_name, [])])
    visited_station = set()
    visited_station.add(start_station_name)
    while open_station:
        current_station_name, path_fr = open_station.popleft()
        path = path_fr.copy()
        path.append(current_station_name)

        if current_station_name == end_station_name:
            return path

        for next_station in map[current_station_name].links:
            if next_station.name not in visited_station:
                open_station.append((next_station.name, path))
                visited_station.add(next_station.name)
    return []


def dfs_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    stack = [start_station]
    came_from = {start_station: None}
    visited = set()  # 用于跟踪已访问的站点

    while stack:
        current = stack.pop()
        visited.add(current)  # 将当前站点标记为已访问

        if current == end_station:
            break

        for neighbor in current.links:
            if neighbor not in came_from and neighbor not in visited:
                stack.append(neighbor)
                came_from[neighbor] = current

    if end_station not in came_from:
        return []  # Path not found

    # Reconstruct path
    path = []
    while current:
        path.append(current.name)
        current = came_from[current]
    return path[::-1]


def dijkstra_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], method='M') -> List[str]:
    end_station = map[end_station_name]
    open_station = [(0, start_station_name)]  # 使用站点名称而非对象
    came_from = {start_station_name: None}
    cost_so_far = {start_station_name: 0}

    while open_station:
        current_cost, current_name = heapq.heappop(open_station)
        current = map[current_name]

        if current == end_station:
            break

        for neighbor in current.links:
            neighbor_name = neighbor.name
            new_cost = current_cost + heuristic(current, neighbor, method)
            if neighbor_name not in cost_so_far or new_cost < cost_so_far[neighbor_name]:
                cost_so_far[neighbor_name] = new_cost
                priority = new_cost
                heapq.heappush(open_station, (priority, neighbor_name))
                came_from[neighbor_name] = current_name

    if end_station_name not in came_from:
        return []

    path = []
    current_name = end_station_name
    while current_name:
        path.append(current_name)
        current_name = came_from[current_name]
    return path[::-1]


def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    return astar_get_path(start_station_name, end_station_name, map)


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
    path1 = astar_get_path(start_station_name, end_station_name, stations)
    # path2 = astar_get_path(start_station_name, end_station_name, stations, method='E')
    path2 = bfs_get_path(start_station_name, end_station_name, stations)
    path3 = dfs_get_path(start_station_name, end_station_name, stations)
    path4 = dijkstra_get_path(start_station_name, end_station_name, stations)
    # path4 = dijkstra_get_path(start_station_name, end_station_name, stations, method='E')

    path = [path1, path2, path3, path4]
    color = ['green', 'yellow', 'blue', 'red']
    name = ['A*', 'BFS', 'DFS', 'Dijkstra']
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines,
              color, name)

    # plot_path([path1], 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines, ["red"], ["A*"])
