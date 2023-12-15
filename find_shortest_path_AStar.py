from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
import math as m

class Node:
    def __init__(self, station, g, h, parent=None):
        self.station = station
        self.g = g
        self.h = h  
        self.parent = parent 

    def f(self):
        return self.g + self.h 
    
    def __lt__(self, other):
        return self.f() < other.f()

def heuristic(station1, station2):
    return (m.sqrt((station1.position[0] - station2.position[0]) ** 2 ) +
            m.sqrt((station1.position[1] - station2.position[1]) ** 2))

def astar(start_station, end_station, map):
    open_set = []
    closed_set = set()

    start_node = Node(start_station, 0, heuristic(start_station, end_station))
    heapq.heappush(open_set, (start_node.f(), start_node))

    while open_set:
        _, current_node = heapq.heappop(open_set)

        if current_node.station == end_station:
            path = []
            while current_node:
                path.insert(0, current_node.station.name)
                current_node = current_node.parent
            return path

        closed_set.add(current_node.station)

        for neighbor_station in current_node.station.links:
            if neighbor_station in closed_set:
                continue

            tentative_g = current_node.g + 1  

            neighbor_node = Node(
                station=neighbor_station,
                g=tentative_g,
                h=heuristic(neighbor_station, end_station),
                parent=current_node
            )

            if any(neighbor_node.station == node.station for _, node in open_set):
                continue

            heapq.heappush(open_set, (neighbor_node.f(), neighbor_node))

    return None  


def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    path = astar(start_station, end_station, map)

    if path:
        print("Shortest path found:", path)
        return path
    else:
        return ["No path found"]


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
