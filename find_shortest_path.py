
# 创建柱状图的示例
#algorithms = ['算法1', '算法2', '算法3']
#execution_times = [time1, time2, time3]

#plt.bar(algorithms, execution_times)
#plt.xlabel('算法')
#plt.ylabel('执行时间（秒）')
#plt.title('执行时间比较')
#plt.show()

from typing import List, Dict, Union
import heapq
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from typing import List
import heapq
class Station:
    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.links = set()
        
def euclidean_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)



def haversine_distance(coord1, coord2):
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    radius = 6371.0

    distance = radius * c
    return distance

def manhattan_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return abs(lat1 - lat2) + abs(lon1 - lon2)

def astar(start_station: Station, end_station: Station, station_map: Dict[str, Station]) -> List[str]:
    open_set = []
    heapq.heappush(open_set, (0, start_station))

    came_from: Dict[Station, Union[None, Station]] = {start_station: None}
    g_score: Dict[Station, float] = {start_station: 0}

    while open_set:
        current_g, current_station = heapq.heappop(open_set)

        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = came_from[current_station]
            return path[::-1]

        for neighbor in current_station.links:
            tentative_g = g_score[current_station] + haversine_distance(current_station.position, neighbor.position)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                # h_score = manhattan_distance(neighbor.position, end_station.position)
                # f_score = tentative_g + h_score
                #h_score = euclidean_distance(neighbor.position, end_station.position)
                #f_score = tentative_g + h_score
                f_score = tentative_g + haversine_distance(neighbor.position, end_station.position)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current_station

    return []  # 如果没有找到路径



def astar_man(start_station: Station, end_station: Station, station_map: Dict[str, Station]) -> List[str]:
    open_set = []
    heapq.heappush(open_set, (0, start_station))

    came_from: Dict[Station, Union[None, Station]] = {start_station: None}
    g_score: Dict[Station, float] = {start_station: 0}

    while open_set:
        current_g, current_station = heapq.heappop(open_set)

        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = came_from[current_station]
            return path[::-1]

        for neighbor in current_station.links:
            tentative_g = g_score[current_station] + manhattan_distance(current_station.position, neighbor.position)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                # h_score = manhattan_distance(neighbor.position, end_station.position)
                # f_score = tentative_g + h_score
                h_score = manhattan_distance(neighbor.position, end_station.position)
                f_score = tentative_g + h_score
                #f_score = tentative_g + haversine_distance(neighbor.position, end_station.position)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current_station

    return []  # 如果没有找到路径

def astar_euc(start_station: Station, end_station: Station, station_map: Dict[str, Station]) -> List[str]:
    open_set = []
    heapq.heappush(open_set, (0, start_station))

    came_from: Dict[Station, Union[None, Station]] = {start_station: None}
    g_score: Dict[Station, float] = {start_station: 0}

    while open_set:
        current_g, current_station = heapq.heappop(open_set)

        if current_station == end_station:
            path = []
            while current_station:
                path.append(current_station.name)
                current_station = came_from[current_station]
            return path[::-1]

        for neighbor in current_station.links:
            tentative_g = g_score[current_station] + euclidean_distance(current_station.position, neighbor.position)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                # h_score = manhattan_distance(neighbor.position, end_station.position)
                # f_score = tentative_g + h_score
                h_score = euclidean_distance(neighbor.position, end_station.position)
                f_score = tentative_g + h_score
                #f_score = tentative_g + haversine_distance(neighbor.position, end_station.position)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current_station

    return []  # 如果没有找到路径


def dijkstra(start_station_name: str, end_station_name: str, station_map: Dict[str, Station]) -> List[str]:
    start_station = station_map[start_station_name]
    end_station = station_map[end_station_name]

    # 初始化距离字典，记录从起始站到每个站点的距离
    distance = {station.name: float('inf') for station in station_map.values()}
    distance[start_station.name] = 0


    # 优先队列，存储 (距离, 站点) 的元组，距离小的先出队
    priority_queue = [(0, start_station.name)]  # 注意这里将站点对象改为站点名字

    # 记录每个站点的前驱站点，用于还原路径
    came_from = {}

    while priority_queue:
        current_distance, current_station_name = heapq.heappop(priority_queue)

        # 如果当前站点已经处理过，跳过
        if current_distance > distance[current_station_name]:
            continue

        current_station = station_map[current_station_name]

        # 遍历当前站点相邻的站点
        for neighbor in current_station.links:
            # 计算新的距离
            new_distance = distance[current_station_name] + haversine_distance(current_station.position, neighbor.position)

            # 如果新的距离比当前记录的距离小，更新距离并加入优先队列
            if new_distance < distance[neighbor.name]:
                distance[neighbor.name] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor.name))
                came_from[neighbor.name] = current_station_name

    # 构造路径
    path = []
    current = end_station.name
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start_station_name)
    return path[::-1]








def compare_all_paths_and_distances(start_stations, end_stations, station_map):
    for start_station in start_stations:
        for end_station in end_stations:
            if start_station != end_station:
                start_station_name = start_station.name
                end_station_name = end_station.name

                # 计算 A* 算法的最短路径
                astar_path = astar(start_station, end_station, station_map)
                astar_distance = calculate_total_distance(astar_path, station_map)

                # 计算 Dijkstra 算法的最短路径
                dijkstra_path = dijkstra(start_station_name, end_station_name, station_map)
                dijkstra_distance = calculate_total_distance(dijkstra_path, station_map)

                # 比较路径和距离
                if astar_path != dijkstra_path or astar_distance != dijkstra_distance:
                    print(f"起点：{start_station_name}, 终点：{end_station_name}")
                    print(f"A* 最短路径：{astar_path}, 距离：{astar_distance} km")
                    print(f"Dijkstra 最短路径：{dijkstra_path}, 距离：{dijkstra_distance} km")
                    print("")

# def compare_distances(all_stations, stations):
#     for start_station in all_stations:
#         for end_station in all_stations:
#             if start_station != end_station:
#                 # 使用曼哈顿距离进行A*算法
#                 manhattan_path = astar_man(start_station, end_station, stations)
#                 manhattan_distance = calculate_total_distance(manhattan_path, stations)

#                 # 使用欧几里得距离进行A*算法
#                 euclidean_path = astar_euc(start_station, end_station, stations)
#                 euclidean_distance = calculate_total_distance(euclidean_path, stations)

#                 if manhattan_path != euclidean_path:
#                     print(f"起点：{start_station.name}, 终点：{end_station.name}")
#                     print(f"曼哈顿距离：{manhattan_distance} km")
#                     print(f"欧几里得距离：{euclidean_distance} km")
#                     print(f"曼哈顿距离和欧几里得距离的差异：{abs(manhattan_distance - euclidean_distance)} km")
#                     print("")

def compare_distances(start_station_name, end_station_name, station_map):
    start_station = station_map[start_station_name]
    end_station = station_map[end_station_name]

    # 使用曼哈顿距离进行A*算法
    manhattan_path = astar_man(start_station, end_station, station_map)
    manhattan_distance = calculate_total_distance(manhattan_path, station_map)

    # 使用欧几里得距离进行A*算法
    euclidean_path = astar_euc(start_station, end_station, station_map)
    euclidean_distance = calculate_total_distance(euclidean_path, station_map)

    print(f"起点：{start_station_name}, 终点：{end_station_name}")
    print(f"A* 曼哈顿距离启发式路径：{manhattan_path}, 曼哈顿距离：{manhattan_distance} km")
    print(f"A* 欧几里得距离启发式路径：{euclidean_path}, 欧几里得距离：{euclidean_distance} km")
    print(f"曼哈顿距离和欧几里得距离的差异：{abs(manhattan_distance - euclidean_distance)} km")
    print("")



def calculate_total_distance(path, station_map):
    total_distance = 0.0
    for i in range(1, len(path)):
        station1 = station_map[path[i-1]]
        station2 = station_map[path[i]]
        distance = haversine_distance(station1.position, station2.position)
        total_distance += distance
    return total_distance


def get_path(start_station_name: str, end_station_name: str, station_map: Dict[str, Station]) -> List[str]:
    start_station = station_map.get(start_station_name)
    end_station = station_map.get(end_station_name)

    if start_station and end_station:
        #return dijkstra(start_station, end_station, station_map)
        return astar_man(start_station, end_station, station_map)
    else:
        print("Invalid start or end station name.")
        return []


 
if __name__ == '__main__':
    start_time = time.time()
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name', metavar='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name', metavar='end_station_name')

    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    total_distance = calculate_total_distance(path, stations)
    
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'C:/Users/lenovo/Desktop/fall 2023/STA303/assign3/visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    end_time = time.time()
    execution_time = end_time - start_time





# # 输出路线和距离
# if __name__ == '__main__':
#     # 获取地铁站点和地铁线路数据
#     stations, _ = build_data()

#     # 假设你要从A站到B站
#     start_station_name = "Acton Town"
#     end_station_name ="Blackwall" #"Dagenham Heathway""Arsenal"

#     start_station = stations[start_station_name]
#     end_station = stations[end_station_name]

#     # 调用A*算法获取路径
#     #path = dijkstra(start_station, end_station, stations)#astar
#     path = astar(start_station, end_station, stations)

#     # 计算实际距离
#     total_distance = calculate_total_distance(path, stations)

#     print(f"最短路径：{path}")
#     print(f"实际距离：{total_distance} km")


# if __name__ == '__main__':
# # 用法示例
#     stations, _ = build_data()
#     all_stations = set(stations.values())
#     compare_all_paths_and_distances(all_stations, all_stations, stations)



# if __name__ == '__main__':
#     stations, _ = build_data()

#     # 假设你选择前五个站点作为起始集合，后五个站点作为目标集合
#     all_stations = list(stations.values())

#     # 比较曼哈顿距离和欧几里得距离
#     compare_distances(all_stations, stations)

# if __name__ == '__main__':
#     stations, _ = build_data()

#     # 假设你选择前五个站点作为起始集合，后五个站点作为目标集合
#     all_stations = list(stations.values())

#     # 比较曼哈顿距离和欧几里得距离
#     compare_distances("Richmond", "Loughton", stations)





