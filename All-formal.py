from typing import List
from plot_underground_path import plot_path  # 导入绘图函数
from build_data import Station, build_data  # 导入数据构建函数
import argparse  # 解析命令行参数的库
import math  # 数学运算库
import heapq  # 堆队列算法的库
import time  # 时间相关的库
from queue import PriorityQueue  # 优先队列的库

# 定义全局变量和初始值
counter = 0
times = 10

# 计算路径总成本的函数
def calculate_total_cost(path: List[str], map: dict[str, Station], heuristic) -> float:
    # 计算路径上每一段的成本总和
    total_cost = 0.0
    for i in range(len(path) - 1):
        station_name1 = path[i]
        station_name2 = path[i + 1]
        station1 = map[station_name1]
        station2 = map[station_name2]
        total_cost += heuristic(station1, station2)  # 使用启发式函数计算成本
    return total_cost

# 节点类，用于表示地铁站
class Node:
    def __init__(self, station, cost_so_far, estimated_total_cost):
        self.station = station
        self.cost_so_far = cost_so_far
        self.estimated_total_cost = estimated_total_cost
    
    def __lt__(self, other):
        return self.estimated_total_cost < other.estimated_total_cost

# 启发式函数：平方根
def heuristic_sqrt(station1, station2):
    # 计算两个站点之间的距离（启发式函数的一种）
    lat1, long1 = station1.position
    lat2, long2 = station2.position
    return math.sqrt((lat2 - lat1)**2 + (long2 - long1)**2)

# 启发式函数：曼哈顿距离
def heuristic_Manhattan(station1, station2):
    lat1, long1 = station1.position
    lat2, long2 = station2.position
    return abs(lat2 - lat1) + abs(long2 - long1)

# 启发式函数：固定值1
def heuristic_1(station1, station2):
    return 1

# Haversine距离计算函数
def haversine_distance(station1, station2):
    lat1, long1 = station1.position
    lat2, long2 = station2.position
    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, long1, lat2, long2])
    # Haversine公式计算距离
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_earth = 6371  # 地球半径，单位为公里
    distance = radius_earth * c
    return distance

# 存储启发式函数的字典
heuristic_functions = {
    heuristic_sqrt: 'heuristic_sqrt',
    heuristic_Manhattan: 'heuristic_Manhattan',
    heuristic_1: 'heuristic_1',
    haversine_distance: 'haversine_distance'
}
############################
heuristic = haversine_distance  # 手动更改启发式函数，这里使用Haversine距离函数
############################


#以下是所有算法实现：A*, UCS, Dijkstra's, Greedy, Bidirectional A*
def get_path_A(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    ##the usage of heuristic function: 
    
    visited = set()
    queue = []
    heapq.heappush(queue, Node(start_station, 0, heuristic(start_station, end_station)))
    parent = {start_station: None}
    cost_so_far = {start_station: 0}
    
    while queue:
        current_node = heapq.heappop(queue)
        current_station = current_node.station
        
        if current_station == end_station:
            path = []
            #write the path lengh
            while current_station:
                path.append(current_station.name)
                current_station = parent[current_station]
            return path[::-1]
        
        visited.add(current_station)
        
        for neighbor_station in current_station.links:
            neighbor = map[neighbor_station.name]
            new_cost = cost_so_far[current_station] + heuristic(current_station, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                total_cost = new_cost + heuristic(neighbor, end_station)
                heapq.heappush(queue, Node(neighbor, new_cost, total_cost))
                parent[neighbor] = current_station
    
    return []

def get_path_Greedy(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    visited = set()
    queue = []
    heapq.heappush(queue, Node(start_station, 0, heuristic(start_station, end_station)))
    parent = {start_station: None}

    while queue:
        current_node = heapq.heappop(queue)
        current_station = current_node.station

        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = parent[current_station]
            return path[::-1]

        visited.add(current_station)

        for neighbor_station in current_station.links:
            neighbor = map[neighbor_station.name]
            if neighbor not in visited:
                total_cost = heuristic(neighbor, end_station)
                heapq.heappush(queue, Node(neighbor, 0, total_cost))
                parent[neighbor] = current_station

    return []

def get_path_Bidirectional(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Data structures for start to end search
    start_visited = set()
    start_queue = []
    heapq.heappush(start_queue, Node(start_station, 0, heuristic(start_station, end_station)))
    start_parent = {start_station: None}
    start_cost_so_far = {start_station: 0}

    # Data structures for end to start search
    end_visited = set()
    end_queue = []
    heapq.heappush(end_queue, Node(end_station, 0, heuristic(end_station, start_station)))
    end_parent = {end_station: None}
    end_cost_so_far = {end_station: 0}

    while start_queue and end_queue:
        # Expand start search
        current_start_node = heapq.heappop(start_queue)
        current_start_station = current_start_node.station

        start_visited.add(current_start_station)

        for neighbor_station in current_start_station.links:
            neighbor = map[neighbor_station.name]
            new_start_cost = start_cost_so_far[current_start_station] + heuristic(current_start_station, neighbor)
            if neighbor not in start_cost_so_far or new_start_cost < start_cost_so_far[neighbor]:
                start_cost_so_far[neighbor] = new_start_cost
                total_start_cost = new_start_cost + heuristic(neighbor, end_station)
                heapq.heappush(start_queue, Node(neighbor, new_start_cost, total_start_cost))
                start_parent[neighbor] = current_start_station

                # Check if a common node is found
                if neighbor in end_visited:
                    merged_path = _merge_paths(start_parent, end_parent, neighbor)
                    
                    return merged_path

        # Expand end search
        current_end_node = heapq.heappop(end_queue)
        current_end_station = current_end_node.station

        end_visited.add(current_end_station)

        for neighbor_station in current_end_station.links:
            neighbor = map[neighbor_station.name]
            new_end_cost = end_cost_so_far[current_end_station] + heuristic(current_end_station, neighbor)
            if neighbor not in end_cost_so_far or new_end_cost < end_cost_so_far[neighbor]:
                end_cost_so_far[neighbor] = new_end_cost
                total_end_cost = new_end_cost + heuristic(neighbor, start_station)
                heapq.heappush(end_queue, Node(neighbor, new_end_cost, total_end_cost))
                end_parent[neighbor] = current_end_station

                # Check if a common node is found
                if neighbor in start_visited:
                    merged_path = _merge_paths(start_parent, end_parent, neighbor)
                    
                    return merged_path

    # Merge paths if a common node is found
    common_node = _find_common_node(start_visited, end_visited)
    if common_node:
        return _merge_paths(start_parent, end_parent, common_node)

    # If no common node found and both searches have ended without meeting,
    # attempt to connect the paths manually
    return _connect_paths(start_parent, end_parent, start_visited, end_visited)

def _connect_paths(start_parent, end_parent, start_visited, end_visited):
    # Attempt to connect paths manually
    for node in start_visited:
        if node in end_visited:
            return _merge_paths(start_parent, end_parent, node)

    # If no common node found, return an empty path
    return []

# Rest of your code (heuristic functions, Node class, etc.) remains unchanged

def _merge_paths_meta(start_parent,end_parent,  intersection_node):
    # Merge paths from start and end searches
    start_path = []
    end_path = []

    # Traverse from intersection node to start node
    while intersection_node:
        start_path.append(intersection_node.name)
        intersection_node = start_parent.get(intersection_node)  # Using .get() to handle None

    # Reverse and store the start path
    start_path = start_path[::-1]

    # Reset intersection_node to the common node
    intersection_node = end_parent.get(intersection_node)

    # Traverse from intersection node to end node
    while intersection_node:
        end_path.append(intersection_node.name)
        intersection_node = end_parent.get(intersection_node)

    # Remove the common node (intersection node) if it appears in the end path
    end_path = end_path[:-1]

    return start_path + end_path[::-1]

def _merge_paths(start_parent,end_parent, intersection_node):
    start_parent_copy = start_parent.copy()
    end_parent_copy = end_parent.copy()
    a = _merge_paths_meta(start_parent,end_parent, intersection_node)[::-1]
    a.pop(0)
    b=_merge_paths_meta(end_parent,start_parent, intersection_node)
#     b.pop(0)
    return b+a


def ucs(graph, home, destination, map):
    """
    Perform Uniform Cost Search on a graph from a start location (home) to a goal location (destination).

    Parameters:
    graph (dict): A dictionary representation of the graph where keys are location names and values
                  are lists of neighbors.
    home (str): The starting location in the graph.
    destination (str): The goal location to reach in the graph.
    map (dict): Dictionary mapping station names to Station objects

    Returns:
    list: A list of locations (str) from 'home' to 'destination' representing the shortest path.
    """

    if home not in graph:
        raise TypeError(str(home) + ' not found in graph!')
    if destination not in graph:
        raise TypeError(str(destination) + ' not found in graph!')

    queue = PriorityQueue()
    queue.put((0, [home]))  # Enqueue with initial cost 0
    visited = set()

    while not queue.empty():
        nowcost, nowPath = queue.get()

        if nowPath[-1] == destination:
            return nowPath  # If the current node is the destination, return the path

        if nowPath[-1] in visited:
            continue  # If the current node has been visited, skip it

        visited.add(nowPath[-1])

        for neighbor_station in graph[nowPath[-1]]:
            if neighbor_station not in visited:
                appendedPath = nowPath[:]  # Clone the old path for the new one
                appendedPath.append(neighbor_station)
                queue.put((heuristic(map[neighbor_station],map[nowPath[-1]])+nowcost, appendedPath))  # Enqueue with a constant cost (0) to explore shortest paths first

    # If the goal is not reachable, return None
    return None

def get_path_UCS(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> list[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    # Convert station links to station names for the graph representation
    graph = {station.name: [neighbor.name for neighbor in station.links] for station in map.values()}

    # Perform UCS
    path = ucs(graph, start_station.name, end_station.name, map)
    
    if path:
        return path  # Return the path from UCS
    
    return []  # Return an empty list if no path is found

class Node_Di:
    def __init__(self, station, cost):
        self.station = station
        self.cost = cost
    
    def __lt__(self, other):
        return self.cost < other.cost
    
def get_path_Dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    visited = set()
    queue = []
    heapq.heappush(queue, Node_Di(start_station, 0))
    parent = {start_station: None}
    cost_so_far = {start_station: 0}
    
    while queue:
        current_node = heapq.heappop(queue)
        current_station = current_node.station
        
        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = parent[current_station]
            return path[::-1]
        
        visited.add(current_station)
        
        for neighbor_station in current_station.links:
            neighbor = map[neighbor_station.name]
            new_cost = cost_so_far[current_station] + heuristic(current_station, neighbor)  # Modify this line if you have specific costs between stations
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(queue, Node_Di(neighbor, new_cost))
                parent[neighbor] = current_station
    
    return []

#主函数
# 主函数，执行路径规划和计时
if __name__ == '__main__':
    with open('output.txt', 'a') as file:
        file.write(f"Heuristic: {heuristic_functions.get(heuristic, 'Unknown heuristic')}\n")
        file.write(f"*********************************\n")

    stations, underground_lines = build_data()  # 构建地铁站和线路数据

    # 需要查询的地铁站对
    station_pairs = [
        # 地铁站对的列表
("Paddington","Shepherd's Bush (C)"),
("East Acton","High Barnet"),
("Farringdon","Edgware"),
("South Kensington","Tower Gateway"),
("St. John's Wood","Park Royal"),
("Sudbury Town","Harrow & Wealdston"),
("Blackhorse Road","Roding Valley"),
("Notting Hill Gate","Holborn"),
("South Ealing","Clapham South"),
("Royal Victoria","Croxley"),
("Marble Arch","West Finchley"),
("Surrey Quays","Hammersmith"),
("West Harrow","Borough"),
("Latimer Road","Chesham"),
("Kilburn Park","Shadwell"),
("Queen's Park","Wembley Central"),
("Latimer Road","Snaresbrook"),
("Park Royal","Colliers Wood"),
("East India","Perivale"),
("Hainault","Temple")
    ]
    # 初始化统计数据
    pair_num = len(station_pairs)
    pair_count = 0
    
    time_UCS = 0
    time_Greedy = 0
    time_A = 0
    time_BidirectA = 0
    time_Dijkstra = 0
    
    path_UCS = 0
    path_Greedy = 0
    path_A = 0
    path_BidirectA = 0
    path_Dijkstra = 0
    
    for start, end in station_pairs:
        pair_count += 1
        # Run your path-finding algorithms for each station pair here
        # Update start_station_name and end_station_name with start and end, then execute the algorithms
        start_station_name = start
        end_station_name = end
        
        #以下是分别运行4个算法得出并写入file：路径长度，运行时间
        
        #####################################################
        # Record the start time
        start_time = time.time()

        counter =0
        
        path = get_path_A(start_station_name, end_station_name, stations)
        path_A += calculate_total_cost(path,stations,heuristic)

        
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        time_A += end_time - start_time
        if(pair_count == pair_num):
            with open('output.txt', 'a') as file:
                file.write(f"A*: {path_A/pair_num:.6f}\n")
                file.write(f"time:{time_A:.6f}\n")
                file.write(f"---------------------------------\n")
        #####################################################
        start_time = time.time()
        counter =0

        path = get_path_UCS(start_station_name, end_station_name, stations)


        # Record the end time
        end_time = time.time()
        path_UCS += calculate_total_cost(path,stations,heuristic)
            # Calculate the elapsed time
#         elapsed_time = end_time - start_time
        time_UCS += end_time - start_time
        if(pair_count == pair_num):
            with open('output.txt', 'a') as file:
                file.write(f"UCS: {path_UCS/pair_num:.6f}\n")

                file.write(f"time:{time_UCS:.6f}\n")
                file.write(f"---------------------------------\n")
        #####################################################
        start_time = time.time()

        path = get_path_Greedy(start_station_name, end_station_name, stations)
        path_Greedy+=calculate_total_cost(path,stations,heuristic)

        # Record the end time
        end_time = time.time()

            # Calculate the elapsed time
#         elapsed_time = end_time - start_time
        time_Greedy += end_time - start_time
        if(pair_count == pair_num):
            with open('Final_Path_length.txt', 'a') as file:
                file.write(f"Greedy: {path_Greedy/pair_num:.6f}\n")
                file.write(f"time:{time_Greedy:.6f}\n")
                file.write(f"---------------------------------\n")
        #####################################################
        start_time = time.time()

        # Your Python code goes here

        path = get_path_Bidirectional(start_station_name, end_station_name, stations)
        path_BidirectA+=calculate_total_cost(path,stations,heuristic)

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
#         elapsed_time = end_time - start_time
        time_BidirectA += end_time - start_time
        if(pair_count == pair_num):
            with open('output.txt', 'a') as file:
                file.write(f"Bidirectional A*: {path_BidirectA/pair_num:.6f}\n")
                file.write(f"time:{time_BidirectA:.6f}\n")
                file.write(f"---------------------------------\n")
                
        #####################################################
        start_time = time.time()

        path = get_path_Dijkstra(start_station_name, end_station_name, stations)
        path_Dijkstra+=calculate_total_cost(path,stations,heuristic)

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
#         elapsed_time = end_time - start_time
        time_Dijkstra += end_time - start_time
        if(pair_count == pair_num):
            with open('output.txt', 'a') as file:
                file.write(f"Dijkstra: {path_Dijkstra/pair_num:.6f}\n")
                file.write(f"time:{time_Dijkstra:.6f}\n")
                file.write(f"---------------------------------\n")
        # visualization the path
#     Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
#     plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
