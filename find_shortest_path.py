from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
from queue import PriorityQueue
import argparse
import math
from collections import deque
import time

def cal_dis(start_station,end_station,mode): #1:Euclidean 2:manhattan 3:chebyshev 4:0
    st_lat, st_lon = start_station.position
    ed_lat, ed_lon = end_station.position
    if(mode==1): return math.sqrt((st_lat - ed_lat)**2 + (st_lon - ed_lon)**2)
    elif(mode==2): return abs(st_lat - ed_lat) + abs(st_lon - ed_lon)
    elif(mode==3): return max(abs(st_lat - ed_lat), abs(st_lon - ed_lon))
    elif(mode==4): return 0
    else:
        print('wrong mode')
        return None
    
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
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    queue = PriorityQueue()
    queue.put((cal_dis(start_station,end_station,4), ([start_station_name],0)))# change the mode
    Close_set=[]#store the name of explored node
    count=0
    while not queue.empty():
            count+=1
            priority, (list_sta,dis2sta) = queue.get()
            last_sta_name=list_sta[-1]
            if last_sta_name==end_station_name:
                return list_sta,dis2sta,count
            Close_set.append(last_sta_name)
            last_sta=map[last_sta_name]
            for new_sta in last_sta.links: 
                if new_sta.name not in Close_set:
                    temp_list=list_sta.copy()
                    temp_list.append(new_sta.name)
                    new_dis2sta=dis2sta+cal_dis(new_sta,last_sta,1)
                    new_priority=new_dis2sta+cal_dis(new_sta,end_station,4)# change the mode
                    queue.put((new_dis2sta+new_priority,(temp_list,new_dis2sta)))
    return None

def dfs(start_station_name: str, end_station_name: str, map: dict[str, Station], visited=None) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    visited = set()
    stack = [(start_station_name, 0, [start_station_name])]
    
    while stack:
        node_name, path_length, path = stack.pop()
        if node_name not in visited:
            visited.add(node_name)
            
            if node_name == end_station_name:
                return path, path_length
            cur_node=map[node_name]
            # Push unvisited neighbors onto the stack with updated path length
            for neighbor in reversed(list(cur_node.links)):
                if neighbor not in visited:
                    new_path = path + [neighbor.name]
                    stack.append((neighbor.name, path_length + cal_dis(map[node_name],neighbor,1), new_path))
                    
def bfs(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    visited = set()
    queue = deque([(start_station_name, 0, [start_station_name])])
    
    while queue:
        node_name, path_length, path = queue.popleft()
        if node_name not in visited:
            visited.add(node_name)
            
            if node_name == end_station_name:
                return path, path_length
            cur_node = map[node_name]
            
            # Enqueue unvisited neighbors with updated path length
            for neighbor in cur_node.links:
                if neighbor.name not in visited:
                    new_path = path + [neighbor.name]
                    queue.append((neighbor.name, path_length + cal_dis(map[node_name], neighbor, 1), new_path))


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
    start_time = time.time()
#     path,length,count = get_path(start_station_name, end_station_name, stations)
#     path,length=dfs(start_station_name, end_station_name, stations)
    path,length=bfs(start_station_name, end_station_name, stations)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
    print(f"The length of path: {length}")
#     print(f"The count sum in the priority queue: {count}")
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
