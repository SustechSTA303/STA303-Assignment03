import time
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
from anytree import Node, RenderTree
import csv

my_dict = {}
with open('london/underground_routes.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)  # 创建 CSV 读取器对象
    for row in csvreader:  # 逐行读取数据
        list = []
        # row 是一个列表，包含当前行的数据
        start_end = row[0] + "-" + row[1]
        value = my_dict.get(start_end, 0)  # 存在返回value，不存在返回0
        if value != 0:
            list = value
            list.append(row[2])
            my_dict[start_end] = list
        else:
            list.append(row[2])
            my_dict[start_end] = list
my_dict.pop('station1-station2')  # 输出：{'11-163': ['1'], '11-212': ['1'], '49-87': ['1', '9'], ......}


def f_score_Euclidean_distance(current, neighbor, g_score, end_station):
    """
    根据欧几里得距离算出来的f_score
    goal：距离最短
    参数：current：当前的站点
         neighbor：相邻的站点
    """
    tentative_g_score = g_score[current.name] + Euclidean_distance(current.name, neighbor.name,
                                                                   stations)  # 前面没两站之间的g_score+这次的

    # h_score：距离终点的代价，启发函数
    h_score = Euclidean_distance(neighbor.name, end_station.name, stations)
    f_score = tentative_g_score + h_score
    return tentative_g_score, f_score


def f_score_NumberOfTransfers():
    """
    根据换乘次数算出来的f_score
    goal：换乘次数最少
    """
    pass


def find_paths(node, target_leaf_name, path=[]):
    """
    node：根节点（Node类型）
    target_leaf_name：输入名称即可
    寻找所有叶子节点为目标的支路并打印路线上的所有节点
    返回：返回一个大list，包含许多小list代表每条路径
    """
    all_paths = []
    path = path + [node.name]
    if node.is_leaf and node.name == target_leaf_name:
        all_paths.append(path.copy())
    for child in node.children:
        all_paths.extend(find_paths(child, target_leaf_name, path))
    return all_paths


def Calculate_distance(path, map: dict[str, Station]) -> List[str]:
    """
    path：一个list，包含该条路线上所有站点Station对象
    goal：给定一条路线，计算该条路线的总距离
    """
    total_distance = 0
    path1 = path[0:len(path) - 1]  # path中第一个到倒数第二个元素
    path2 = path[1:len(path)]  # path中第二个到最后一个元素
    for i in range(len(path1)):  # len(path1) = len(path2)
        station_name_1 = path1[i].name
        station_name_2 = path2[i].name
        distance = Euclidean_distance(station_name_1, station_name_2, stations)
        total_distance = total_distance + distance
    return total_distance


def Euclidean_distance(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    创建一个叫Euclidean_distance方法：
    goal：计算任意两站点之间的欧几里得距离
    input：任意两个站点的name（Station.name）
    output：根据position（坐标）计算出的两站点之间的欧几里得距离
    """
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    return ((start_station.position[0] - end_station.position[0]) ** 2 +
            (start_station.position[1] - end_station.position[1]) ** 2) ** 0.5


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
    start_station = map[start_station_name]  # 是Station类型
    end_station = map[end_station_name]
    # 从这开始 ####################################################################################################
    nodes = {}  # 用于存储站点名称与对应Node对象的映射
    root = Node(start_station)
    nodes[start_station.name] = root
    '''
    start_station.links是一个set集合，无法切片，我们选择遍历获取其中的元素。start_station.links中每个元素都是一个Station对象
    看看起始和终止点的邻居
    links_start_station = []
    links_end_station = []
    for item in start_station.links:
        links_start_station.append(item.name)
    for item in end_station.links:
        links_end_station.append(item.name)
    print("start_station的邻居为：", links_start_station, "end_station的邻居为：", links_end_station)
    
    start_station.position的数据类型是tuple
    '''
    # 实现A*搜索算法 ##############################################################################################
    open_set = []  # 初始化一个空的优先队列
    heapq.heappush(open_set, (0, start_station))  # 将起始站点添加到优先队列
    close_set = []  # 用于记录路径
    # 遍历all_stations列表中的每一个元素，然后把每一个元素作为键（key），对应的初始值设置为无穷大（float('inf')）。
    g_score = {station: float('inf') for station in stations}  # 初始化所有站点的g_score为无限大-inf
    # print(g_score)-output:{'Acton Town': inf, 'Aldgate': inf,......}
    g_score[start_station.name] = 0  # 起始站点的g_score为0
    # 一直执行，直到open_set变为空
    while open_set:
        # 前面定义的是(0, start_station)，所以返回[1]代表Station对象。[0]返回的是当前的优先级
        current = heapq.heappop(open_set)[1]  # 从优先队列中取出当前站点，Station对象
        if current == end_station:  # 如果当前站点是目标站点，重构并返回路径
            """
            # 打印树
            # for pre, fill, node in RenderTree(root):
            #     print(f"{pre}{node.name}")
            path = []
            target_node = nodes[current.name]  # 找到目标节点的所有父节点
            ancestors = target_node.ancestors
            for ancestor in ancestors:
                # type(ancestor)-<class 'anytree.node.node.Node'>; type(ancestor.name))-<class 'build_data.Station'>
                path.append(ancestor.name.name)
            path.append(current.name)  # 把终点站加进去
            return path  # 返回path
            """
            continue
        else:  # 如果当前站点不是终点
            close_set.append(current)
            for neighbor in current.links:  # 遍历当前与当前Station对象相邻的所有Station对象
                # 遍历当前站点的所有邻接站点
                if neighbor in close_set:
                    continue
                else:
                    ###############################################################################
                    nodes[neighbor.name] = Node(neighbor, parent=nodes[current.name])
                    ###############################################################################
                    # tentative暂定的；tentative_g_score = 该站点前一个站点的g_socre + 该站点到上一个站点的g_socre
                    # tentative_g_score = g_score[current.name] + Euclidean_distance(current.name, neighbor.name,
                    #                                                                stations)  # 前面没两站之间的g_score+这次的
                    # 计算f_score = g(n)（是节点n距离起点的代价） + h(n)（节点n距离终点的预计代价，这也就是A*算法的启发函数）#######
                    tentative_g_score, f_score = f_score_Euclidean_distance(current, neighbor, g_score, end_station)
                    #################################################################################################
                    if tentative_g_score < g_score[neighbor.name]:  # 确保不会回去
                        # 如果从当前站点到邻接站点的g_score更小
                        # close_set[neighbor] = current
                        g_score[neighbor.name] = tentative_g_score
                        # f_score = g(n)（是节点n距离起点的代价） + h(n)（节点n距离终点的预计代价，这也就是A*算法的启发函数）
                        # f_score = tentative_g_score + Euclidean_distance(neighbor.name, end_station.name, stations)
                        heapq.heappush(open_set, (f_score, neighbor))  # 将邻接站点加入优先队列
    # 打印树
    # for pre, fill, node in RenderTree(root):
    #     print(f"{pre}{node.name}")
    paths = find_paths(root, end_station)
    list_distance = []  # 用于储存每条路径的距离
    for path in paths:
        distance = Calculate_distance(path, stations)  # 计算该条路径的距离
        list_distance.append(distance)
    min_index = list_distance.index(min(list_distance))
    print(len(list_distance), min_index)
    return paths[min_index-1]  # while结束，返回一个包含所有路线的大list，大list包含许多小list，每个小list是一条路线
    # 到此结束 ####################################################################################################


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
    # print(stations)-返回{'Acton Town': <build_data.Station object at 0x000001915B0B7760>,......}
    # 计时开始##############################################################################################
    start_time = time.time()
    ################################################### ####################################################
    path = get_path(start_station_name, end_station_name, stations)
    # 记录结束时间############################################################################################
    end_time = time.time()
    # 计算经过的时间
    elapsed_time = end_time - start_time
    print(f"The elapsed time is {elapsed_time} seconds.")
    ########################################################################################################

    # test #################################################################################################
    path_name = [station.name for station in path]
    print("The path form", start_station_name, "to", end_station_name, "is", path_name)
    distance = Calculate_distance(path, stations)
    print("The total distance of this path is:", distance)
    # over #################################################################################################

    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path_name, 'visualization_underground/my_shortest_path_in_London_railway.html', stations,
              underground_lines)