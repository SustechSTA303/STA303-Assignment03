from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
from queue import PriorityQueue
from queue import Queue
import time

def compute_distance(path, map):
    distance = 0
    for i in range(len(path) - 1):
        current_station = map[path[i]]
        next_station = map[path[i + 1]]
        distance += current_station.get_distance(next_station)
    return distance

def get_path_a_star(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # A* algorithm implementation
    open_set = [start_station]
    came_from = {}
    g_score = {start_station: 0}
    
    # 使用三个字典分别存储使用不同启发函数的 f_score
    f_score_euclidean = {start_station: start_station.get_heuristic_euclidean(end_station)}
    f_score_manhattan = {start_station: start_station.get_heuristic_manhattan(end_station)}
    f_score_zero = {start_station: 0}
    
    while open_set:
        current_station = min(open_set, key=lambda station: f_score_euclidean[station])
        if current_station == end_station:
            # Reconstruct path
            path = []
            while current_station in came_from:
                path.insert(0, current_station.name)
                current_station = came_from[current_station]
            path.insert(0, start_station.name)
            return path

        open_set.remove(current_station)
        for neighbor in current_station.get_neighbors():
            tentative_g_score = g_score[current_station] + current_station.get_distance(neighbor)

            # 计算使用欧几里得距离的 f_score
            f_score_euclidean[neighbor] = tentative_g_score + neighbor.get_heuristic_euclidean(end_station)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_station
                g_score[neighbor] = tentative_g_score
                if neighbor not in open_set:
                    open_set.append(neighbor)

            # 计算使用曼哈顿距离的 f_score
            f_score_manhattan[neighbor] = tentative_g_score + neighbor.get_heuristic_manhattan(end_station)

            # 使用0作为启发函数
            f_score_zero[neighbor] = 0
            
    return []

def get_path_dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Dijkstra's algorithm implementation
    open_set = PriorityQueue()
    open_set.put((0, start_station))
    came_from = {}
    cost_so_far = {start_station: 0}

    while not open_set.empty():
        current_cost, current_station = open_set.get()

        if current_station == end_station:
            # Reconstruct path
            path = []
            while current_station in came_from:
                path.insert(0, current_station.name)
                current_station = came_from[current_station]
            path.insert(0, start_station.name)
            return path

        for neighbor in current_station.get_neighbors():
            new_cost = cost_so_far[current_station] + current_station.get_distance(neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current_station
                open_set.put((new_cost, neighbor))

    return []

def get_path_bfs(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Breadth-First Search implementation
    visited = set()
    queue = Queue()
    queue.put(start_station)
    visited.add(start_station)
    
    parent_dict = {start_station: None}

    while not queue.empty():
        current_station = queue.get()
        current_parent = parent_dict[current_station]
        
        if current_station == end_station:
            # Reconstruct path
            path = [end_station.name]
            while current_station != start_station:
                path.insert(0, current_parent.name)
                current_station = parent_dict[current_parent]
            path.insert(0, start_station.name)
            return path

        for neighbor in current_station.get_neighbors():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)
                # Store the parent station for path reconstruction
                parent_dict[neighbor] = current_station


    return []


def get_path_dfs(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # DFS algorithm implementation
    visited = set()
    shortest_path = []

    def dfs(current_station, current_path, current_cost):
        nonlocal shortest_path
        visited.add(current_station)

        if current_station == end_station:
            # Reconstruct path
            if not shortest_path or current_cost < shortest_path[1]:
                shortest_path = (current_path.copy(), current_cost)
            return

        for neighbor in current_station.get_neighbors():
            if neighbor not in visited:
                new_path = current_path + [neighbor.name]
                new_cost = current_cost + current_station.get_distance(neighbor)
                dfs(neighbor, new_path, new_cost)

    start_time = time.time()

    dfs(start_station, [start_station.name], 0)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"DFS Running time: {elapsed_time} s")

    return shortest_path[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    stations, underground_lines = build_data()

    # A* Algorithm
    start_time11 = time.time()
    
    path_astar_euclidean = get_path_a_star(start_station_name, end_station_name, stations)
    print(f'A* (Euclidean) Shortest path: {path_astar_euclidean}')
    
    end_time11 = time.time()
    elapsed_time11 = end_time11 - start_time11
    print(f"A* (Euclidean) Running time: {elapsed_time11} s")
    
    distance_astar_euclidean = compute_distance(path_astar_euclidean, stations)
    print(f'A* (Euclidean) Distance: {distance_astar_euclidean} km')
    #Manhattan
    start_time12 = time.time()
    path_astar_manhattan = get_path_a_star(start_station_name, end_station_name, stations)
    print(f'A* (Manhattan) Shortest path: {path_astar_manhattan}')
    
    end_time12 = time.time()
    elapsed_time12 = end_time12 - start_time12
    print(f"A* (Manhattan) Running time: {elapsed_time12} s")
    
    distance_astar_manhattan = compute_distance(path_astar_manhattan, stations)
    print(f'A* (Manhattan) Distance: {distance_astar_manhattan} km')
    
    #Zero
    start_time13 = time.time()
    path_astar_zero = get_path_a_star(start_station_name, end_station_name, stations)
    print(f'A* (Zero) Shortest path: {path_astar_zero}')
    end_time13 = time.time()
    elapsed_time13 = end_time13 - start_time13
    print(f"A* (Zero) Running time: {elapsed_time13} s")
    
    distance_astar_zero = compute_distance(path_astar_zero, stations)
    print(f'A* (Zero) Distance: {distance_astar_zero} km')


    # Dijkstra's Algorithm
    start_time2 = time.time()
    
    path_dijkstra = get_path_dijkstra(start_station_name, end_station_name, stations)
    print(f'Dijkstra Shortest path: {path_dijkstra} ')
    
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    print(f"Dijkstra Running time: {elapsed_time2} s")
    
    distance_dijkstra = compute_distance(path_dijkstra, stations)
    print(f'Dijkstra Distance: {distance_dijkstra} km')

    # Deepth-First Search
    start_time3 = time.time()
    
    path_dfs = get_path_dfs(start_station_name, end_station_name, stations)
    print(f'DFS Shortest path: {path_dfs}')
    
    end_time3 = time.time()
    elapsed_time3 = end_time3 - start_time3
    print(f"DFS Running time: {elapsed_time2} s")
    
    distance_dfs = compute_distance(path_dfs, stations)
    print(f'DFS Distance: {distance_dfs} km')
    
    # Breadth-First Search
#    start_time3 = time.time()
    
#    path_bfs = get_path_bfs(start_station_name, end_station_name, stations)
#    print(f'BFS Shortest path: {path_bfs}')
    
#    end_time3 = time.time()
#    elapsed_time3 = end_time3 - start_time3
#    print(f"BFS Running time: {elapsed_time3} s")


    # Visualization
    if path_astar_euclidean:
        plot_path(path_astar_euclidean, 'visualization_underground/my_astar_path_euclidean.html', stations, underground_lines)
    if path_astar_manhattan:
        plot_path(path_astar_manhattan, 'visualization_underground/my_astar_path_manhattan.html', stations, underground_lines)
    if path_astar_zero:
        plot_path(path_astar_zero, 'visualization_underground/my_astar_path_zero.html', stations, underground_lines)

        
    if path_dijkstra:
        plot_path(path_dijkstra, 'visualization_underground/my_dijkstra_path.html', stations, underground_lines)
    if path_dfs:
        plot_path(path_dfs, 'visualization_underground/my_dfs_path.html', stations, underground_lines)
#    if path_bfs:
#        plot_path(path_bfs, 'visualization_underground/my_bfs_path.html', stations, underground_lines)
