from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
import math as m


class AStarNode:
    def __init__(self, station, g, h, parent=None):
        self.station = station
        self.g = g  # Actual cost from start to current node
        self.h = h  # Heuristic cost from current node to goal
        self.f = g + h  # Total cost
        self.parent = parent  # Parent node

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal_node):
    
    lat1, lon1 = node.position
    lat2, lon2 = goal_node.position
    return (m.sqrt((lat1 - lat2)**2) + m.sqrt((lon1 - lon2)**2))


def astar_search(start_node, goal_node):
    open_set = [AStarNode(start_node, 0, heuristic(start_node, goal_node))]
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.station == goal_node:
            path = []
            while current_node:
                path.insert(0, current_node.station.name)
                current_node = current_node.parent
            return path

        closed_set.add(current_node.station)

        for neighbor in current_node.station.links:
            if neighbor not in closed_set:
                g = current_node.g + 1  # Assuming equal cost for all connections
                h = heuristic(neighbor, goal_node)
                f = g + 0.1 * h
                heapq.heappush(open_set, AStarNode(neighbor, g, h, current_node))

    return None


def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Run A* search
    path = astar_search(start_station, end_station)

    if path:
        print("Shortest path found:", path)
    else:
        print("Path not found.")

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

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
