from collections import deque
import heapq
import math
import random
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import platform
import time
import pandas as pd


# distance measures. A and B are Station objects for the two stations, return "distance" which is in km.
def d(A, B):
    latA, lonA, latB, lonB = (A.position[0], A.position[1], B.position[0], B.position[1])
    ra = 6378140  # 赤道半径
    rb = 6356755  # 极半径
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = math.radians(latA)
    radLonA = math.radians(lonA)
    radLatB = math.radians(latB)
    radLonB = math.radians(lonB)
    
    pA = math.atan(rb / ra * math.tan(radLatA))
    pB = math.atan(rb / ra * math.tan(radLatB))
    x = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(radLonA - radLonB))
    c1 = (math.sin(x) - x) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(x / 2) ** 2
    c2 = (math.sin(x) + x) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(x / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    distance = round(distance / 1000, 4)
    return distance


def Haversine(A, B):
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    latA, lonA, latB, lonB = map(math.radians, [A.position[0], A.position[1], 
                                                B.position[0], B.position[1]])

    # Differences in coordinates
    dlat = latA - latB
    dlon = lonA - lonB

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(latA) * math.cos(latB) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

def Manhattan(A, B):
    latA, lonA, latB, lonB = map(math.radians, [A.position[0], A.position[1], 
                                                B.position[0], B.position[1]])
    third_point= Station(None, None, [latA, lonB])
    distance = Haversine(A, third_point) + Haversine(B, third_point)
    return distance



def Euclidean(A, B):
    latA, lonA, latB, lonB = map(math.radians, [A.position[0], A.position[1], 
                                                B.position[0], B.position[1]])
    third_point= Station(None, None, [latA, lonB])
    distance = math.sqrt(Haversine(A, third_point)**2 + Haversine(B, third_point)**2)
    return distance



def L_infty(A, B):
    latA, lonA, latB, lonB = map(math.radians, [A.position[0], A.position[1], 
                                                B.position[0], B.position[1]])
    third_point= Station(None, None, [latA, lonB])
    distance = max(Haversine(A, third_point), Haversine(B, third_point))
    return distance





# "start" and "end" are Station objects. "map" maps a station name to the station.
def BFS(start, end, map):
    visited = set()
    queue = deque([(start, [start.name])])  
        # Each element in the queue is a tuple (current_node, path_so_far)

    while queue:
        current_station, path = queue.popleft()

        if current_station.name == end.name:
            total_distance = 0
            # Iterate over pairs of consecutive elements
            for i in range(len(path) - 1):
                total_distance += Haversine(map[path[i]], map[path[i + 1]])
            return [path, total_distance]  # Return the path if the target is reached

        if current_station.name not in visited:
            visited.add(current_station.name)

            for neighbor in current_station.links:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor.name]))

    return None  # Return None if no path is found




# "start" and "end" are Station objects.
def Bellman_Ford(start, end):
    nodes = set()
    found_nodes = set()
    queue = deque([start])

    while queue:
        found_station = queue.popleft()
        if found_station.name not in found_nodes:
            found_nodes.add(found_station.name)
            nodes.add(found_station)
            for neighbor in found_station.links:
                if neighbor not in found_nodes:
                    queue.append(neighbor)
    
    costs = {node.name: float('inf') for node in nodes}
    costs[start.name] = 0
    paths = {node.name: [] for node in nodes}
    paths[start.name] = [start.name]

    for _ in range(len(nodes) - 1):
        for node in nodes:
            for neighbor in node.links:
                if costs[node.name] + Haversine(node, neighbor) < costs[neighbor.name]:
                    costs[neighbor.name] = costs[node.name] + Haversine(node, neighbor)
                    paths[neighbor.name] = paths[node.name] + [neighbor.name]
                # For undirected graphs, consider adding the reverse edge as well
                if costs[neighbor.name] + Haversine(node, neighbor) < costs[node.name]:
                    costs[node.name] = costs[neighbor.name] + Haversine(node, neighbor)
                    paths[node.name] = paths[neighbor.name] + [node.name]

    return [paths[end.name], costs[end.name]]



# "start" and "end" are Station objects.
def Dijkstra(start, end):
    priority_queue = [(0, start, [])]  # (distance, node, path)
    visited = set()
    distances = {start: 0}
    paths = {start.name: []}

    while priority_queue:
        distance, current_node, path = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if not path:
            path = [current_node.name]
        else:
            path = path + [current_node.name]

        if current_node == end:
            return [path, distances[end]] 

        for neighbor in current_node.links:
            if neighbor not in visited:
                new_distance = distances[current_node] + Haversine(current_node, neighbor)

                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    paths[neighbor.name] = path

                    heapq.heappush(priority_queue, (new_distance, neighbor, path))

    return float('inf'), []  # If the target is not reachable




# "start" and "end" are Station objects. "h" is the heuristic distance function.
def Astar(start, end, h):
    open_set = [(0 + h(start, end), 0, start, [])]  # (f, g, node, path), "f" is the total cost and "g" is the actual cost
                                                    # f = g + h 
    closed_set = set()

    while open_set:
        _, g, current_node, path = heapq.heappop(open_set)

        if current_node == end:
            return [path + [current_node.name], g]

        if current_node in closed_set:
            continue

        closed_set.add(current_node)

        for neighbor in current_node.links:
            if neighbor not in closed_set:
                new_g = g + Haversine(current_node, neighbor)
                heapq.heappush(open_set, (new_g + h(neighbor, end), new_g, neighbor, path + [current_node.name]))

    return [[], float('inf')] # If the target is not reachable




# Implement the following function









# "alg" is the chosen searching algorithm, and "heur" is the string indicating the heuristic function when A* is chosen.
def get_path(alg, start_station_name: str, end_station_name: str, map: dict[str, Station], heur = None) -> List[str]:
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

    if alg == "DFS":
        shortest = Dijkstra(start_station, end_station)

    if alg == "DFS_forced":
        shortest = Dijkstra(start_station, end_station)

    if alg == "BFS":
        shortest = BFS(start_station, end_station, map)

    if alg == "Bellman_Ford":
        shortest = Bellman_Ford(start_station, end_station)

    if alg == "Dijkstra":
        shortest = Dijkstra(start_station, end_station)

    if alg == "Astar":
        if heur == "Haversine":
            shortest = Astar(start_station, end_station, Haversine)
        if heur == "Manhattan":
            shortest = Astar(start_station, end_station, Manhattan)
        if heur == "Euclidean":
            shortest = Astar(start_station, end_station, Euclidean)
        if heur == "L_infty":
            shortest = Astar(start_station, end_station, L_infty)
    return shortest



## (start, end) sampling method

def generate_random_pairs(nodes, pairs_size):
    all_pairs = [(start, target) for start in nodes for target in nodes]
    random_pairs = random.sample(all_pairs, pairs_size)
    return random_pairs


if __name__ == '__main__':

    # 创建ArgumentParser对象
    #parser = argparse.ArgumentParser()
    # 添加命令行参数
    #parser.add_argument('start_station_name', type=str, help='start_station_name')
    #parser.add_argument('end_station_name', type=str, help='end_station_name')
    #args = parser.parse_args()
    #start_station_name = args.start_station_name
    #end_station_name = args.end_station_name

    #start_station_name = "Acton Town"
    #end_station_name = "Beckton Park"
    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    searching_algs = ["BFS", "Bellman_Ford", "Dijkstra", "Astar"]
    heuristic_funcs = ["Haversine", "Manhattan", "Euclidean", "L_infty"]

    start_end_pairs = generate_random_pairs(stations.keys(), 450)
    df_path_lengths = pd.DataFrame()
    df_alg_times = pd.DataFrame()
    
    for alg in searching_algs:
        paths, path_lengths = [], []
        T1 = time.perf_counter()
        for travel in start_end_pairs:
            start_station_name, end_station_name = travel[0], travel[1]
            path, path_length = get_path(alg, start_station_name, end_station_name, stations, "Haversine")
            paths.append(path)
            path_lengths.append(path_length)
        T2 = time.perf_counter()
        df_path_lengths[alg] = path_lengths
        run_time = ((T2 - T1)*1000)
        df_alg_times[alg] = [run_time]
        print(alg + ': %s (ms)' % run_time)

    df_path_lengths.to_csv("comparison_data/path_lengths.csv", index=False)

    df_path_lengths_astar = pd.DataFrame()
    for h in heuristic_funcs:
        paths, path_lengths = [], []
        T1 = time.perf_counter()
        for travel in start_end_pairs:
            start_station_name, end_station_name = travel[0], travel[1]
            path, path_length = get_path("Astar", start_station_name, end_station_name, stations, h)
            paths.append(path)
            path_lengths.append(path_length)
        T2 = time.perf_counter()
        df_path_lengths_astar[h] = path_lengths
        run_time = ((T2 - T1)*1000)
        df_alg_times[h] = [run_time]
        print(h + ': %s (ms)' % run_time)
    
    df_path_lengths_astar.to_csv("comparison_data/path_lengths_astar.csv", index=False)
    df_alg_times.to_csv("comparison_data/alg_times.csv", index=False)
    print("OK")
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red

    # plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)





        
        