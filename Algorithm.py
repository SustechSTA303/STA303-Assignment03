from typing import List, Tuple
from queue import PriorityQueue
from build_data import Station
import math
import time
from sys import getsizeof
from collections import deque

# haversine distance
def haversine_distance(start_station, end_station):
    lat1, lon1 = start_station.position
    lat2, lon2 = end_station.position
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# manhattan distance
def manhattan_distance(start_station, end_station):
    lat1, lon1 = start_station.position
    lat2, lon2 = end_station.position
    return abs(lat2 - lat1) + abs(lon2 - lon1)

# euclidean distance
def euclidean_distance(start_station, end_station):
    lat1, lon1 = start_station.position
    lat2, lon2 = end_station.position
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def chebyshev_distance(start_station, end_station):
    lat1, lon1 = start_station.position
    lat2, lon2 = end_station.position
    delta_lat = abs(lat2 - lat1)
    delta_lon = abs(lon2 - lon1)
    return max(delta_lat, delta_lon)

def astar_algorithm(start_station_name, end_station_name, map, distance_function):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    if distance_function == 'haversine':
        distance_fc = haversine_distance
    elif distance_function == 'manhattan':
        distance_fc = manhattan_distance
    elif distance_function == 'euclidean':
        distance_fc = euclidean_distance
    elif distance_function == 'chebyshev':
        distance_fc = chebyshev_distance
    else:
        raise ValueError("Invalid distance function")
        
    start_time = time.time()
    open_set = PriorityQueue()
    total_distance = 0
    expanded_nodes = 0

    open_set.put((0, start_station))
    came_from = {}
    cost_so_far = {}

    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0
    while not open_set.empty():
        current = open_set.get()[1]
        # Increment the counter when a node is expanded
        expanded_nodes += 1

        if current == end_station:
            break
        for next in current.links:
            new_cost = cost_so_far[current.name] + distance_fc(next, current)
            if next.name not in cost_so_far or new_cost < cost_so_far[next.name]:
                cost_so_far[next.name] = new_cost
                priority = new_cost + distance_fc (next, end_station)
                open_set.put((priority, next))
                came_from[next.name] = current.name
    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Expanded nodes:', expanded_nodes)
    path = []
    current = end_station_name
    while current != start_station_name:
        path.append(current)
        current = came_from[current]
    path.append(start_station_name)
    path.reverse()
    total_distance = calculate_total_distance(path,map)
    return path, total_distance


def dijkstra_algorithm(start_station_name, end_station_name, map):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    total_distance = 0
    expanded_nodes = 0

    start_time = time.time()
    open_set = PriorityQueue()

    open_set.put((0, start_station))
    came_from = {}
    cost_so_far = {}

    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0

    while not open_set.empty():
        current = open_set.get()[1]
        expanded_nodes += 1
        if current == end_station:
            break
        for next_station in current.links:
            new_cost = cost_so_far[current.name] + haversine_distance(current, next_station)

            if next_station.name not in cost_so_far or new_cost < cost_so_far[next_station.name]:
                cost_so_far[next_station.name] = new_cost
                priority = new_cost
                open_set.put((priority, next_station))
                came_from[next_station.name] = current.name
    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Expanded nodes:', expanded_nodes)
    
    path = []
    current = end_station_name
    while current != start_station_name:
        path.append(current)
        current = came_from[current]

    path.append(start_station_name)
    path.reverse()
    total_distance = calculate_total_distance(path,map)
    return path,total_distance


def spfa_algorithm(start_station_name, end_station_name, map):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    total_distance = 0

    start_time = time.time()
    queue = deque([start_station])
    in_queue = {station: False for station in map.values()}
    in_queue[start_station] = True

    distance = {station: float('inf') for station in map.values()}
    distance[start_station] = 0

    came_from = {station: None for station in map.values()}

    # Initialize the counter for expanded nodes
    expanded_nodes = 0

    while queue:
        current = queue.popleft()
        in_queue[current] = False

        for next_station in current.links:
            new_distance = distance[current] + haversine_distance(current, next_station)

            if new_distance < distance[next_station]:
                distance[next_station] = new_distance
                came_from[next_station] = current

                if not in_queue[next_station]:
                    queue.append(next_station)
                    in_queue[next_station] = True

                    # Increment the counter when a node is put into the queue
                    expanded_nodes += 1

    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Expanded nodes:', expanded_nodes)

    path = []
    current = end_station
    while current != start_station:
        path.append(current.name)
        current = came_from[current]

    path.append(start_station_name)
    path.reverse()
    total_distance = calculate_total_distance(path, map)
    return path, total_distance


def bellman_ford_algorithm(start_station_name, end_station_name, map):
    # Check if the start and end stations exist in the map
    total_distance = 0
    start_time = time.time()
    if start_station_name not in map or end_station_name not in map:
        raise ValueError("Start or end station not found in the map.")

    # Initialize distance and predecessor dictionaries
    distance = {station_name: float('inf') for station_name in map.keys()}
    predecessor = {station_name: None for station_name in map.keys()}

    # Set the distance from the start station to itself as 0
    distance[start_station_name] = 0

    # Initialize the counter for expanded edges
    expanded_edges = 0

    # Relax edges repeatedly to find the shortest paths
    for _ in range(len(map) - 1):
        for station_name, station in map.items():
            for neighbor in station.links:
                new_distance = distance[station_name] + haversine_distance(station, neighbor)
                if new_distance < distance[neighbor.name]:
                    distance[neighbor.name] = new_distance
                    predecessor[neighbor.name] = station_name

                    # Increment the counter when an edge is relaxed
                    expanded_edges += 1

    # Check for negative cycles
    for station_name, station in map.items():
        for neighbor in station.links:
            if distance[station_name] + haversine_distance(station, neighbor) < distance[neighbor.name]:
                raise ValueError("Graph contains a negative cycle.")

    # Reconstruct the shortest path
    path = reconstruct_path(start_station_name, end_station_name, predecessor)
    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Expanded Notes:', expanded_edges)
    
    total_distance = calculate_total_distance(path,map)
    return path,total_distance

def reconstruct_path(start_station_name, end_station_name, predecessor):
    path = []
    current_station = end_station_name
    while current_station is not None:
        path.insert(0, current_station)
        current_station = predecessor[current_station]
    return path


def ucs_algorithm(start_station_name, end_station_name, map):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    total_distance = 0

    # Initialize the counter for expanded nodes
    expanded_nodes = 0

    start_time = time.time()
    priority_queue = PriorityQueue()
    priority_queue.put((0, start_station))
    visited = set([start_station_name])
    cost_so_far = {start_station_name: 0}
    came_from = {start_station_name: None}

    while not priority_queue.empty():
        current_cost, current_station = priority_queue.get()

        if current_station == end_station:
            end_time = time.time()
            print('Time:', end_time - start_time)
            print('Expanded nodes:', expanded_nodes)
            # Reconstruct the path
            path = []
            while current_station is not None:
                path.insert(0, current_station.name)
                current_station = came_from[current_station.name]
            total_distance = calculate_total_distance(path,map)
            return path, total_distance

        for neighbor in current_station.links:
            new_cost = cost_so_far[current_station.name] + haversine_distance(current_station, neighbor)
            if neighbor.name not in visited or new_cost < cost_so_far[neighbor.name]:
                visited.add(neighbor.name)
                cost_so_far[neighbor.name] = new_cost
                priority = new_cost
                priority_queue.put((priority, neighbor))
                came_from[neighbor.name] = current_station

                # Increment the counter when a node is put into the queue
                expanded_nodes += 1
                
    return [], 0

def calculate_total_distance(path,map):
    total_distance = 0
    for i in range(len(path)-1):
        total_distance += haversine_distance(map[path[i]],map[path[i+1]])
    return total_distance