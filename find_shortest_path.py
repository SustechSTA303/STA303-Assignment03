from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
import math

# Implement the following function

def heuristic(station1, station2):
    #dx = abs(station1.position[0] - station2.position[0])
    #dy = abs(station1.position[1] - station2.position[1])
    #return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    #return max(abs(station1.position[0] - station2.position[0]), abs(station1.position[1] - station2.position[1]))
    
    #return abs(station1.position[0] - station2.position[0]) + abs(station1.position[1] - station2.position[1])
    return math.sqrt((station1.position[0] - station2.position[0])**2 + (station1.position[1] - station2.position[1])**2)

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    open_set = []
    closed_set = set()
    came_from = {}
    
    heapq.heappush(open_set, (0, start_station))
    came_from[start_station] = None

    while open_set:
        current_cost, current_station = heapq.heappop(open_set)

        closed_set.add(current_station)

        for neighbor in current_station.links:
            if neighbor in closed_set:
                continue

            tentative_cost = current_cost + 1 + heuristic(neighbor, end_station)

            neighbor_in_open_set = [(cost,s) for cost, s in open_set if s == neighbor]
            if not neighbor_in_open_set or tentative_cost < neighbor_in_open_set[0][0]:
                heapq.heappush(open_set, (tentative_cost, neighbor))
                came_from[neighbor] = current_station

        if current_station == end_station:
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = came_from[current_station]
            return path

    return []
    #add————————
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    pass


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




