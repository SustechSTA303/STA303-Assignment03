import heapq
from build_data import Station


def astar(start_station: Station, end_station: Station, map: dict[str, Station]) -> list:
    distance = euclidean
    heuristic = manhattan

    closed_set = set()
    closed_set.add(start_station)
    start_station.g_score = 0
    start_station.f_score = heuristic(start_station, end_station)
    start_station.come_from = None
    open_set = [(start_station.f_score, start_station)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current.id == end_station.id:
            return reconstruct_path(current)
        for neighbor in current.links:
            if neighbor not in closed_set:
                neighbor.g_score = current.g_score + distance(current, neighbor)
                neighbor.f_score = neighbor.g_score + heuristic(neighbor, end_station)
                neighbor.come_from = current
                heapq.heappush(open_set, (neighbor.f_score, neighbor))
                closed_set.add(neighbor)
    raise Exception("No valid path found by A* Search")


def greedy_bfs(start_station: Station, end_station: Station, map: dict[str, Station]) -> list:
    heuristic = manhattan

    closed_set = set()
    closed_set.add(start_station)
    start_station.score = heuristic(start_station, end_station)
    start_station.come_from = None
    open_set = [(start_station.score, start_station)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current.id == end_station.id:
            return reconstruct_path(current)
        for neighbor in current.links:
            if neighbor not in closed_set:
                neighbor.score = heuristic(neighbor, end_station)
                neighbor.come_from = current
                heapq.heappush(open_set, (neighbor.score, neighbor))
                closed_set.add(neighbor)
    raise Exception("No valid path found by Greedy Best First Search")


def dijkstra(start_station: Station, end_station: Station, map: dict[str, Station]) -> list:
    distance = euclidean

    closed_set = set()
    closed_set.add(start_station)
    start_station.g_score = 0
    start_station.come_from = None
    open_set = [(start_station.g_score, start_station)]
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current.id == end_station.id:
            return reconstruct_path(current)
        for neighbor in current.links:
            if neighbor not in closed_set:
                neighbor.g_score = current.g_score + distance(current, neighbor)
                neighbor.come_from = current
                heapq.heappush(open_set, (neighbor.g_score, neighbor))
                closed_set.add(neighbor)
    raise Exception("No valid path found by Dijkstra's Algorithm")


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
    return abs(a.position[0] - b.position[0]) + abs(a.position[1] - b.position[1])

def euclidean(a: Station, b: Station) -> float:
    """
    Calculate the eulidean distance between two stations
    Args:
        a(Station): The first station
        b(Station): The second station
    Returns:
        float: The distance between two stations
    """
    return ((a.position[0] - b.position[0]) ** 2 + (a.position[1] - b.position[1]) ** 2) ** 0.5
