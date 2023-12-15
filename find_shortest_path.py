from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
from queue import PriorityQueue
from collections import deque
import time
import random

class Node:
    
    def __init__(self,station,parent = None,cost = 0,f = 0):
        self.station = station
        self.parent = parent
        self.cost = cost
        self.f = f
        
    def __lt__(self,other):
        return (self.f + self.cost) < (other.f + other.cost)

def Haversine(lat1,lon1,lat2,lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    R = 6371
    distance = R * c
    return distance

def Manhattan(lat1,lon1,lat2,lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    return Haversine(lat1,0,lat2,0) + Haversine(0,lon1,0,lon2)


def Chebyshev_distance(lat1,lon1,lat2,lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    R = 6371
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    lat_dis = dlat * R
    lon_dis = dlon * R
    return max(lat_dis,lon_dis)

    
    
def BFS(start_node,end_node,heuristic = None):
    count = 0
    visited = set()
    queue = deque([start_node])
    while queue:
        count = count + 1
        node = queue.popleft()
        
        if node.station == end_node.station:
            path = [node.station.name]
            while node.parent is not None:
                node = node.parent
                path.append(node.station.name)
#            print(f'BFS {count}')
            return path[::-1],count
            
        if node.station not in visited:
            visited.add(node.station)
            for adj in node.station.links:
                if adj not in visited:
                    adj_node = Node(adj,node)
                    queue.append(adj_node)
    return None
                
    
def Dijkstra(start_node,end_node,heuristic):
    count = 0
    open_set = PriorityQueue()## node
    close_set = set()## station
    open_set.put(start_node)
    while not open_set.empty():
        count=count+1
        current_node = open_set.get()
        
        #find the end point and return the list
        if current_node.station == end_node.station:
            path = [current_node.station.name]
            while current_node.parent is not None:
                current_node = current_node.parent
                path.append(current_node.station.name)
#            print(f'Dijkstra {count}')
            return path[::-1],count
        
        if current_node.station in close_set:
            continue
            
        close_set.add(current_node.station)
            
        for adj in current_node.station.links:
            if adj not in close_set:
                lat1,lon1 = adj.position
                lat2,lon2 = current_node.station.position
                cost = current_node.cost + Haversine(lat1,lon1,lat2,lon2)
                adj_node = Node(adj,current_node,cost)
                open_set.put(adj_node)
                
    return None

def Greedy_BestFS(start_node,end_node,heuristic):
    count = 0
    visited = set()
    pq = PriorityQueue()
    pq.put(start_node)
    while not pq.empty():
        count = count+1
        
        current_node = pq.get()
        
        if(current_node.station == end_node.station):
            path = [current_node.station.name]
            while current_node.parent is not None:
                current_node = current_node.parent
                path.append(current_node.station.name)
#            print(f'UCS {count}')
            return path[::-1],count

        if current_node.station in visited:
            continue
        visited.add(current_node.station)
            
        for adj in current_node.station.links:
            if adj not in visited:
                lat1,lon1 = current_node.station.position
                lat2,lon2 = adj.position
                h = Haversine(lat1,lon1,lat2,lon2)
                new_node = Node(adj,current_node,0,h)
                pq.put(new_node)
                    
    return None
            
            
    


def Astar(start_node,end_node,heuristic):
    count = 0
    open_set = PriorityQueue()## node
    close_set = set()## station
    open_set.put(start_node)
    while not open_set.empty():
        count = count+1
        current_node = open_set.get()
        
        #find the end point and return the list
        if current_node.station == end_node.station:
            path = [current_node.station.name]
            while current_node.parent is not None:
                current_node = current_node.parent
                path.append(current_node.station.name)
#            print(f'Astar {count} with {heuristic.__name__}')
            return path[::-1],count
        
        if current_node.station in close_set:
            continue
            
        close_set.add(current_node.station)
        for adj in current_node.station.links:
            if adj not in close_set:
                lat1,lon1 = adj.position
                lat2,lon2 = current_node.station.position
                cost = current_node.cost + Haversine(lat1,lon1,lat2,lon2)
                lat3,lon3 = end_node.station.position
                f = heuristic(lat1,lon1,lat3,lon3)+cost
                adj_node = Node(adj,current_node,cost,f)
                open_set.put(adj_node)
                
    return None

    

# Implement the following function
def get_path(method,start_station_name: str, end_station_name: str, map: dict[str, Station],heuristic = None):
    start_station = map[start_station_name] ##station
    start_node = Node(start_station,None,0,0)
    end_station = map[end_station_name]
    end_node = Node(end_station,None,0,0)
    return method(start_node,end_node,heuristic)
    

    
if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    
    methods = [Astar,BFS,Dijkstra,Greedy_BestFS]
    heuristics = [Haversine,Manhattan,Chebyshev_distance]
    
    stations, underground_lines,stations_name = build_data()
    
    test_list = []#creat a test list
    random.seed(111)
    for i in range(1000):
        index1, index2 = random.sample(range(len(stations_name)),2)
        test_list.append([stations_name[index1],stations_name[index2]])
    for method in methods:
        
        if(method == Astar):
            for heuristic in heuristics:
                total_count = 0
                start_time = time.time()
                for i in range(len(test_list)):
                    path,count = get_path(method,test_list[i][0], test_list[i][1], stations,heuristic)
                    total_count = total_count + count
                end_time = time.time()
                print(f'use: {method.__name__} with heuristic function: {heuristic.__name__} cost: {end_time-start_time:4f}s total node that test: {total_count}')
        else:
            total_count = 0
            start_time = time.time()
            for i in range(len(test_list)):
                path,count = get_path(method,test_list[i][0], test_list[i][1], stations,heuristic)
                total_count = total_count + count
            end_time = time.time()
            print(f'use: {method.__name__} cost: {end_time-start_time:4f}s total node that test: {total_count}')
            
    start_time = time.time()    
    path,count = get_path(Astar,start_station_name, end_station_name, stations,Haversine)
    end_time = time.time()
    plot_path(path, 'visualization_underground/Astar_Haver_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f'A star with heuristic function: Euler total count: {count} with time: {end_time - start_time:4f}s')
    start_time = time.time()
    path,count = get_path(Astar,start_station_name, end_station_name, stations,Chebyshev_distance)
    end_time = time.time()
    plot_path(path, 'visualization_underground/Astar_Cheby_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f'A star with heuristic function: Chebyshev distance total count: {count} with time: {end_time - start_time:4f}s')
    start_time = time.time()
    path,count = get_path(Astar,start_station_name, end_station_name, stations,Manhattan)
    end_time = time.time()
    plot_path(path, 'visualization_underground/Astar_Manha_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f'A star with heuristic function: Manhattan distance total count: {count} with time: {end_time - start_time:4f}s')
    start_time = time.time()
    path,count = get_path(BFS,start_station_name,end_station_name,stations)
    end_time = time.time()
    plot_path(path, 'visualization_underground/BFS_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f'BFS total count: {count} with time: {end_time - start_time:4f}s')
    start_time = time.time()
    path,count = get_path(Dijkstra,start_station_name,end_station_name,stations)
    end_time = time.time()
    plot_path(path, 'visualization_underground/Dijkstra_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f'Dijkstra total count: {count} with time: {end_time - start_time:4f}s')
    start_time = time.time()
    path,count = get_path(Greedy_BestFS,start_station_name,end_station_name,stations)
    end_time = time.time()
    plot_path(path, 'visualization_underground/Greed_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(f'Greeedy_BestFS total count: {count} with time: {end_time - start_time:4f}s')    
