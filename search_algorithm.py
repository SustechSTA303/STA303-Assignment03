from queue import PriorityQueue
from build_data import Station, build_data
from collections import deque
import distance

# A* algorithm implementation
def a_star(start_station: Station, end_station: Station, actual_cost_estimate: str, heuristic_cost_estimate: str, map: dict[str,Station]):
    actual_cost_estimate, heuristic_cost_estimate = distance.specify(actual_cost_estimate, heuristic_cost_estimate)
    open_set = PriorityQueue()
    
    open_set.put((0, start_station))  # Priority queue with initial node
    came_from = {}  # Dictionary to store the parent node of each node
    # Cost from start along the best-known path, set maximum inicially, maximum is equivalent to not visited
    g_score = {station: float('inf') for station in map.values()}  
    g_score[start_station] = 0
 
    while not open_set.empty():
        current_cost, current_station = open_set.get()

        if current_station == end_station:
            # Reconstruct the path
            path = []
            while current_station in came_from:
                path.append(current_station.name)
                current_station = came_from[current_station]
            path.append(start_station.name)
            return path[::-1]

        for neighbor in current_station.links:
            tentative_g_score = g_score[current_station] + actual_cost_estimate(current_station, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_station
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_cost_estimate(neighbor, end_station)
                open_set.put((f_score, neighbor))

    return []  # No path found

# uniform cost search algorithm implementation
def uniform_cost(start_station: Station, end_station: Station, actual_cost_estimate: str, map: dict[str,Station]):
    actual_cost_estimate, _ = distance.specify(actual_cost_estimate, 0)
        
    # Priority queue to store nodes with their cumulative cost
    frontier = PriorityQueue()
    frontier.put((0, start_station))  # (cumulative_cost, station)

    # Dictionary to store visited stations and their cumulative costs
    visited = {start_station: 0}

    # Dictionary to store the parent station for each station in the path
    parent = {start_station: None}

    while not frontier.empty():
        cumulative_cost, current_station = frontier.get()

        if current_station == end_station:
            # Reconstruct the path
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = parent[current_station]
            return path[::-1]

        for neighbor in current_station.links:
            # Calculate the cumulative cost to reach the neighbor
            tentative_cost = visited[current_station] + actual_cost_estimate(current_station, neighbor)

            if neighbor not in visited or tentative_cost < visited[neighbor]:
                visited[neighbor] = tentative_cost
                parent[neighbor] = current_station
                frontier.put((tentative_cost, neighbor))
    
    # If the loop completes without finding the end station, there is no path
    return []

# Greedy BFS implementation
def greedy_bfs(start_station: Station, end_station: Station, heuristic_cost_estimate: str, map: dict[str,Station]):
    _, heuristic_cost_estimate = distance.specify(0, heuristic_cost_estimate)
        
    if not start_station or not end_station:
        raise ValueError("Start or end station not found in the station map.")

    # Priority queue to store nodes to be processed based on heuristic
    priority_queue = PriorityQueue()
    priority_queue.put((0, start_station))

    # Dictionary to store visited stations and their parent stations
    visited = {start_station: None}

    while not priority_queue.empty():
        _, current_station = priority_queue.get()

        if current_station == end_station:
            # Reconstruct the path
            path = []
            while current_station:
                path.insert(0, current_station.name)
                current_station = visited[current_station]
            return path

        for neighbor in current_station.links:
            if neighbor not in visited:
                visited[neighbor] = current_station
                priority = heuristic_cost_estimate(neighbor, end_station)
                priority_queue.put((priority, neighbor))

    # If the loop completes without finding the end station, there is no path
    return []
