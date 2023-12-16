from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue
from collections.abc import Iterable
import math
import time
import heapq


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def d(neighbor, station):
    lat1, lon1 = station.position
    lat2, lon2 = neighbor.position
    return haversine(lat1, lon1, lat2, lon2)

def ucs(graph, home, destination):
    if home not in graph:
        raise TypeError(str(home) + ' not found in graph!')
    if destination not in graph:
        raise TypeError(str(destination) + ' not found in graph!')

    queue = PriorityQueue()
    queue.put((0, [home]))
    visited = set()

    while not queue.empty():
        cost, path = queue.get()
        current = path[-1]

        if current == destination:
            return (cost, path)
        
        if current not in visited:
            visited.add(current)
            for neighbor, neighbor_cost in graph[current].items():
                if neighbor not in visited:
                    total_cost = cost + neighbor_cost
                    queue.put((total_cost, path + [neighbor]))
    
    return None

def get_path_ucs(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    # Convert the map to a format that the ucs function can understand
    graph = {station_name: {neighbor.name: d(neighbor, station) for neighbor in station.links} for station_name, station in map.items()}

    # Run the ucs function
    start_time = time.time()
    result = ucs(graph, start_station_name, end_station_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if result is None:
        print("No path found from", start_station_name, "to", end_station_name)
        return None

    cost, path = result
    print("Path found with total cost", cost, "km")
    print("Elapsed time of Uniform Cost Search: ", elapsed_time, "seconds")
    return path, elapsed_time

def heuristic1(a, b):
    (x1, y1) = a.position
    (x2, y2) = b.position
    return abs(x1 - x2) + abs(y1 - y2)

def get_path_A_star1(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_time = time.time()
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    queue = []
    heapq.heappush(queue, (0, start_station))

    came_from = {}
    cost_so_far = {}
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    while queue:
        _, current = heapq.heappop(queue)

        if current == end_station:
            break

        for neighbor in current.links:
            new_cost = cost_so_far[current] + d(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic1(end_station, neighbor)
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time of A*: ", elapsed_time, "seconds")
    
    if end_station not in came_from:
        return None

    current = end_station
    path = []
    while current is not None:
        path.append(current.name)
        current = came_from[current]
    path.reverse()

    return path, elapsed_time

def heuristic2(a, b):
    lat1, lon1 = a.position
    lat2, lon2 = b.position
    return haversine(lat1, lon1, lat2, lon2)

def get_path_A_star2(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_time = time.time()
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    queue = []
    heapq.heappush(queue, (0, start_station))

    came_from = {}
    cost_so_far = {}
    came_from[start_station] = None
    cost_so_far[start_station] = 0

    while queue:
        _, current = heapq.heappop(queue)

        if current == end_station:
            break

        for neighbor in current.links:
            new_cost = cost_so_far[current] + d(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic2(end_station, neighbor)
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time of A*: ", elapsed_time, "seconds")
    
    if end_station not in came_from:
        return None

    current = end_station
    path = []
    while current is not None:
        path.append(current.name)
        current = came_from[current]
    path.reverse()

    return path, elapsed_time


