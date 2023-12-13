from math import sqrt
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
from queue import PriorityQueue, Queue
from bezier_util import generate_bezier, bezier_curve_length
import argparse


def get_distance(pos1, pos2, metric = "Euclidean", cp_dict = {}, line_number = None):
    if metric == "Bezier":
        if (line_number, pos1, pos2) in cp_dict:
            return bezier_curve_length(pos1, cp_dict[(line_number, pos1, pos2)], pos2)
    if metric == "Manhattan":
        return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])
    return sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

def is_transfer(sta1: Station, sta2: Station, sta3: Station):
    """
    Return True if sta1 -> sta2 -> sta3 is within the same line.
    """
    set1 = {l[1] for l in sta1.links}
    set2 = {l[1] for l in sta2.links}
    set3 = {l[1] for l in sta3.links}
    return len(set1.intersection(set2.intersection(set3))) != 0

def get_path(
    start_station_name: str,
    end_station_name: str,
    map: dict[str, Station],
    underground_lines: dict,
    algorithm: str = "Astar",
    metric: str = "Euclidean",
    penalty: float = 0,
    cp_dict: dict = {}
) -> List[str]:
    """
    Runs specified algorithm on the map, find the shortest path between a and b
    Args:
        start_station_name(str): The name of the starting station
        end_station_name(str): str The name of the ending station
        map(dict[str, Station]): Mapping between station names and station objects of the name,
                                 Please refer to the relevant comments in the build_data.py
                                 for the description of the Station class
        underground_lines(dict[int, dict]): Line Number -> Underground Line Dict 
        algorithm(str): ["Astar" | "Dijkstra" | "SPFA"]
        metric(str): ["Euclidean" | "Manhattan" | "Bezier"]
        penalty(float): Time estimation for line transfer
        cp_dict(dict[(int, str, str), (float, float)]): 
            (Line Number, Station 1 Name, Station 2 Name) -> Control point position
            Station 1 and Station 2 should be adjacent.
    Returns:
        List[Station]: A path composed of a series of station_name
    """
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
#     print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
#     print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    q = PriorityQueue() if algorithm not in ["SPFA"] else Queue()
    q.put((0, 0, [start_station_name]))
    cur_res = {start_station_name: (0, [start_station_name])}
    vis = set()
    
    while not q.empty():
        
        astar_dis, dis, cur_pth = q.get()
        
        if cur_pth[-1] == end_station_name:
            if algorithm in ["Astar", "Dijkstra"]:
                print(algorithm, metric, penalty)
                print(dis, cur_pth)
                print()
                return cur_pth
        
        if cur_pth[-1] in vis or cur_res[cur_pth[-1]][0] < dis:
            continue
            
        cur_station = map[cur_pth[-1]]
        vis.add(cur_pth[-1])
        
        for sta, tar_line_id in cur_station.links:
            tar_line_id = int(tar_line_id)
            tar_line = underground_lines[tar_line_id]["name"]
            tmp_dis = get_distance(cur_station.position, sta.position, metric, cp_dict, tar_line_id)
            dest_dis = get_distance(end_station.position, sta.position, metric, cp_dict, tar_line_id)
            
            if algorithm != "Astar":
                dest_dis = 0
                
            tmp_dis += penalty / 4 \
                if len(cur_pth) == 1 or is_transfer(map[cur_pth[-2]], map[cur_pth[-1]], sta) \
                else penalty
            
            if sta.name in cur_res and dis + tmp_dis > cur_res[sta.name][0]:
                continue
                
            if sta.name in vis and dis + tmp_dis < cur_res[sta.name][0]:
                vis.remove(sta.name)
                
            cur_res[sta.name] = (dis + tmp_dis, cur_pth + [sta.name])
            
            q.put((dis + tmp_dis + dest_dis, dis + tmp_dis, cur_pth + [sta.name]))
    
    print(algorithm, metric, penalty)
    print(cur_res[end_station_name])
    print()
    return cur_res[end_station_name][1]


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
    cp_dict = generate_bezier(underground_lines)
#     path = get_path(start_station_name, end_station_name, stations)
    
    algorithms = ["Astar", "Dijkstra", "SPFA"]
    metrics = ["Euclidean", "Manhattan", "Bezier"]
    penalties = [0, 0.3]
    
    for algorithm in algorithms:
        for metric in metrics:
            for penalty in penalties:
                if penalty != 0 and algorithm != "Astar":
                    continue
                path = get_path(start_station_name, end_station_name, stations, underground_lines, algorithm, metric, penalty, cp_dict)
                plot_path(path, f"visualization_underground/{start_station_name}_{end_station_name}_{algorithm}_{metric}_{'n' if penalty == 0 else 'p'}.html", stations, underground_lines, cp_dict, metric)
    
