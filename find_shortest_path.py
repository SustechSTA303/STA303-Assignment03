import csv
import os
from queue import PriorityQueue
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from collections import deque


class Node:
    use_default_comparison = True

    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        if Node.use_default_comparison:
            return (self.cost + 2 * self.heuristic) < (other.cost + 2 * other.heuristic)
        else:
            return (self.cost + 0.1 * self.heuristic) < (other.cost + 0.1 * other.heuristic)


def heuristic(state, goal_state):
    return standard_distance(state, goal_state)


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
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    return adjustAstar(start_station, end_station, map)


def Astar(start_station, end_station, my_map):
    frontier = PriorityQueue()
    start_node = Node(state=start_station, cost=0, heuristic=heuristic(start_station, end_station))
    frontier.put(start_node)
    explored = set()
    while True:
        current_node = frontier.get()
        if current_node.state.id == end_station.id:
            return_path = []
            while True:
                return_path.append(current_node.state.name)
                current_node = current_node.parent
                if current_node is None:
                    break
            cost = compute_cost(return_path, my_map)

            return return_path[::-1]
        else:
            explored.add(current_node.state.name)
            for neighbor in get_neighbors(current_node.state):  # 获取当前节点的相邻节点
                if neighbor not in explored:
                    neighbor_node = Node(state=my_map[neighbor], parent=current_node,
                                         cost=current_node.cost + heuristic(current_node.state, my_map[neighbor]),  #
                                         heuristic=heuristic(my_map[neighbor], end_station))
                    frontier.put(neighbor_node)


def adjustAstar(start_station, end_station, my_map):
    frontier = PriorityQueue()
    start_node = Node(state=start_station, cost=0, heuristic=heuristic(start_station, end_station))
    frontier.put(start_node)
    explored = set()
    all_path = {}
    count = 0
    former = ""
    while True:
        count += 1
        if count == 4:
            Node.use_default_comparison = False
            count = 0
        else:
            Node.use_default_comparison = True
        current_node = frontier.get()
        if current_node.state.name == former:
            break
        former = current_node.state.name
        if current_node.state.id == end_station.id:
            return_path = []
            while True:
                return_path.append(current_node.state.name)
                current_node = current_node.parent
                if current_node is None:
                    break
            changedList = return_path[::-1]
            cost = compute_cost(changedList, my_map)
            if changedList[-1] == end_station.name:
                if tuple(changedList) not in all_path:
                    all_path[tuple(changedList)] = cost
            Node.use_default_comparison = False
            if len(all_path) == 10:
                break
        else:
            explored.add(current_node.state.name)
            for neighbor in get_neighbors(current_node.state):  # 获取当前节点的相邻节点
                if neighbor not in explored:
                    neighbor_node = Node(state=my_map[neighbor], parent=current_node,
                                         cost=current_node.cost + heuristic(current_node.state, my_map[neighbor]),  #
                                         heuristic=heuristic(my_map[neighbor], end_station))
                    frontier.put(neighbor_node)
    minPath = min(all_path, key=all_path.get)
    imList = []
    for i in minPath:
        imList.append(i)
    return imList  # 没有找到路径


def BFS(start_station, end_station, my_map):
    visited = set()
    start_node = Node(state=start_station, cost=0, heuristic=heuristic(start_station, end_station))
    queue = deque([start_node])  # 队列中存储车站和路径
    while True:
        if not queue:
            break
        current_node = queue.popleft()
        visited.add(current_node.state.name)
        if current_node.state.id == end_station.id:
            return_path = []
            while True:
                return_path.append(current_node.state.name)
                current_node = current_node.parent
                if current_node is None:
                    break
            changedList = return_path[::-1]
            return changedList
        else:
            visited.add(current_node.state.name)
            for neighbor in get_neighbors(current_node.state):
                if neighbor not in visited:
                    if heuristic(my_map[neighbor], end_station) < current_node.heuristic:
                        neighbor_node = Node(state=my_map[neighbor], parent=current_node,
                                             cost=current_node.cost + heuristic(current_node.state, my_map[neighbor]),
                                             heuristic=heuristic(my_map[neighbor], end_station))

                        queue.append(neighbor_node)

    return []  # 没有找到路径


def adjustBFS(start_station, end_station, my_map):
    former = ""
    visited = set()
    all_path = {}
    count = 0
    start_node = Node(state=start_station, cost=0, heuristic=heuristic(start_station, end_station))
    queue = deque([start_node])  # 队列中存储车站和路径

    while True:
        if not queue:
            break
        current_node = queue.popleft()
        visited.add(current_node.state.name)

        if current_node.state.name == former:
            if len(all_path) != 0:

                break
        former = current_node.state.name
        if current_node.state.id == end_station.id:
            return_path = []
            while current_node is not None:
                return_path.append(current_node.state.name)
                current_node = current_node.parent
            changedList = return_path[::-1]
            cost = compute_cost(changedList, my_map)
            if changedList[-1] == end_station.name:
                if tuple(changedList) not in all_path:
                    all_path[tuple(changedList)] = cost
                    count += 1
        else:
            visited.add(current_node.state.name)
            for neighbor in get_neighbors(current_node.state):
                if neighbor not in visited:
                    if heuristic(my_map[neighbor], end_station) < current_node.heuristic:
                        neighbor_node = Node(state=my_map[neighbor], parent=current_node,
                                             cost=current_node.cost + heuristic(current_node.state, my_map[neighbor]),
                                             heuristic=heuristic(my_map[neighbor], end_station))
                        temp = Node(state=current_node.state, parent=current_node.parent,cost=current_node.cost,heuristic=current_node.heuristic)
                        while temp.parent is not None:
                            if neighbor in get_neighbors(temp.parent.state):
                                neighbor_node.parent = temp.parent
                                break
                            else:
                                temp = temp.parent
                        queue.append(neighbor_node)
        if count == 3:
            break
    minPath = min(all_path, key=all_path.get)
    imList = []
    for i in minPath:
        imList.append(i)
    return imList  # 没有找到路径


def remove_middle_elements(lst, i, j):
    del lst[i:j]
    return lst


def compute_cost(initial, my_map):
    cost = 0
    for i in range(len(initial) - 1):
        cost += standard_distance(my_map[initial[i]], my_map[initial[i + 1]])
    return cost


def get_routes():
    rootDir = os.path.dirname(__file__)
    r = csv.reader(open(os.path.join(rootDir, 'london/underground_routes.csv')))
    neighbours = {}
    names = get_names()
    next(r)  # jump the first line
    for id1, id2, _ in r:
        id1 = int(id1)
        id2 = int(id2)
        if id1 not in neighbours:
            neighbours[id1] = []
            neighbours[id1].append(names[id2])
        else:
            if names[id2] not in neighbours[id1]:
                neighbours[id1].append(names[id2])
        if id2 not in neighbours:
            neighbours[id2] = []
            neighbours[id2].append(names[id1])
        else:
            if names[id1] not in neighbours[id2]:
                neighbours[id2].append(names[id1])
    return neighbours


def get_names():
    rootDir = os.path.dirname(__file__)
    r = csv.reader(open(os.path.join(rootDir, 'london/underground_stations.csv')))
    next(r)
    names = {}
    for record in r:
        ids = int(record[0])
        name = record[3]
        names[ids] = name
    return names


def get_neighbors(station):
    neighbors = get_routes()[station.id]
    return neighbors


def standard_distance(station1, station2):
    return (station1.position[0] - station2.position[0]) ** 2 + (station1.position[1] - station2.position[1]) ** 2


def manhattan_distance(station1, station2):

    distance1 = abs(station1.position[0] - station2.position[0])
    distance2 = abs(station1.position[1] - station2.position[1])

    return distance1 + distance2


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
