import heapq
from build_data import Station
from geopy.distance import geodesic
from math import cos


def astar(start_station: Station, end_station: Station, heuristic: str, alpha=1) -> (list, float, int):
    distance = geodesic_distance

    heuristic = choose_heuristic(heuristic)

    closed_set = set()
    closed_set.add(start_station)
    start_station.g_score = 0
    start_station.f_score = alpha * heuristic(start_station, end_station)
    start_station.come_from = None
    open_set = [(start_station.f_score, start_station)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current.id == end_station.id:
            return reconstruct_path(current), current.g_score, len(closed_set)
        for neighbor in current.links:
            if neighbor not in closed_set:
                neighbor.g_score = current.g_score + distance(current, neighbor)
                neighbor.f_score = neighbor.g_score + alpha * heuristic(neighbor, end_station)
                neighbor.come_from = current
                heapq.heappush(open_set, (neighbor.f_score, neighbor))
                closed_set.add(neighbor)
            else:
                if neighbor.g_score > current.g_score + distance(current, neighbor):
                    neighbor.g_score = current.g_score + distance(current, neighbor)
                    neighbor.f_score = neighbor.g_score + alpha * heuristic(neighbor, end_station)
                    neighbor.come_from = current
                    heapq.heappush(open_set, (neighbor.f_score, neighbor))
    raise Exception("No valid path found by A* Search")


def greedy_bfs(start_station: Station, end_station: Station, heuristic) -> (list, float, int):
    distance = geodesic_distance

    heuristic = choose_heuristic(heuristic)

    closed_set = set()
    closed_set.add(start_station)
    start_station.g_score = 0
    start_station.score = heuristic(start_station, end_station)
    start_station.come_from = None
    open_set = [(start_station.score, start_station)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current.id == end_station.id:
            return reconstruct_path(current), current.g_score, len(closed_set)
        for neighbor in current.links:
            if neighbor not in closed_set:
                neighbor.g_score = current.g_score + distance(current, neighbor)
                neighbor.score = heuristic(neighbor, end_station)
                neighbor.come_from = current
                heapq.heappush(open_set, (neighbor.score, neighbor))
                closed_set.add(neighbor)
    raise Exception("No valid path found by Greedy Best First Search")


def dijkstra(start_station: Station, end_station: Station) -> (list, float, int):
    distance = geodesic_distance

    closed_set = set()
    closed_set.add(start_station)
    start_station.g_score = 0
    start_station.come_from = None
    open_set = [(start_station.g_score, start_station)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current.id == end_station.id:
            return reconstruct_path(current), current.g_score, len(closed_set)
        for neighbor in current.links:
            if neighbor not in closed_set:
                neighbor.g_score = current.g_score + distance(current, neighbor)
                neighbor.come_from = current
                heapq.heappush(open_set, (neighbor.g_score, neighbor))
                closed_set.add(neighbor)
            else:
                if neighbor.g_score > current.g_score + distance(current, neighbor):
                    neighbor.g_score = current.g_score + distance(current, neighbor)
                    neighbor.come_from = current
                    heapq.heappush(open_set, (neighbor.g_score, neighbor))
    raise Exception("No valid path found by Dijkstra's Algorithm")


def choose_heuristic(heuristic: str) -> callable:
    if heuristic == 'manhattan':
        return manhattan
    elif heuristic == 'euclidean':
        return euclidean
    elif heuristic == 'diagonal':
        return diagonal_distance
    else:
        raise Exception("Invalid heuristic")

def reconstruct_path(current: Station) -> list:
    total_path = [current.name]
    while current.come_from:
        current = current.come_from
        total_path.append(current.name)
    return total_path[::-1]


def manhattan(a: Station, b: Station) -> float:
    """
    Calculate the manhattan distance between two stations
    Args:
        a(Station): The first station
        b(Station): The second station
    Returns:
        float: The distance between two stations
    """
    dx = abs(a.position[0] - b.position[0])
    dy = abs(a.position[1] - b.position[1])
    return abs(dx * 111) + abs(dy * 111)

def euclidean(a: Station, b: Station) -> float:
    """
    Calculate the eulidean distance between two stations
    Args:
        a(Station): The first station
        b(Station): The second station
    Returns:
        float: The distance between two stations
    """
    dx = abs(a.position[0] - b.position[0])
    dy = abs(a.position[1] - b.position[1])
    return ((dx * 111) ** 2 + (dy * 111) ** 2) ** 0.5

def geodesic_distance(a: Station, b: Station) -> float:
    """
    Calculate the geodesic distance between two stations
    Args:
        a(Station): The first station
        b(Station): The second station
    Returns:
        float: The distance between two stations
    """
    return geodesic(a.position, b.position).km

def diagonal_distance(a: Station, b: Station) -> float:
    """
    Calculate the diagonal distance between two stations
    Args:
        a(Station): The first station
        b(Station): The second station
    Returns:
        float: The distance between two stations
    """
    dx = abs(a.position[0] - b.position[0])
    dy = abs(a.position[1] - b.position[1])
    return (dx * 111) + (dy * 111) + min(dx, dy) * 111 * (2 ** 0.5 - 2)


