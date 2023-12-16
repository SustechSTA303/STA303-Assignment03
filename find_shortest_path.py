from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
from queue import PriorityQueue
import argparse
import time
from geopy.distance import geodesic

def get_path_Dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]) ->tuple[float, int, float, List[str]]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    cost_so_far=dict()
    cost_so_far[start_station_name]=0

    queue = PriorityQueue()
    queue.put((0,[start_station_name]))
    
    count=0
    start_time=time.time()
    while not queue.empty():
        cost, lst = queue.get()
        current_name=lst[len(lst)-1]
        if end_station_name not in lst:
            count+=1
            linked = map[current_name].links
            for station in linked:
                new_cost=cost_so_far[current_name]+geodesic(map[current_name].position,station.position).kilometers
                if station.name not in cost_so_far or new_cost<cost_so_far[station.name]:
                    cost_so_far[station.name]=new_cost
                    f=new_cost
                    path=lst.copy()
                    path.append(station.name)
                    queue.put((f,path))
        else:
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost, lst
    return None

def get_path_greedyBFS(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> tuple[float, int, float, List[str]]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    queue = PriorityQueue()
    queue.put((0,([start_station.name],0)))
    
    count=0
    start_time=time.time()
    while not queue.empty():
        priority, lst = queue.get()
        path=lst[0]
        cost=lst[1]
        if end_station.name not in path:
            count+=1
            linked = map[path[len(path)-1]].links
            for station in linked:
                new_cost=cost+geodesic(map[path[len(path)-1]].position,station.position).kilometers
                if station.name not in path:
                    path_copy=path.copy()
                    path_copy.append(station.name)
                    priority=geodesic(map[path[len(path)-1]].position,end_station.position).kilometers
                    queue.put((priority,(path_copy,new_cost)))
        else:
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost, path
    return None

def get_path_BellmanFord(start_station_name:str,end_station_name:str,map:dict[str,Station])->tuple[float, int, float, List[str]]:
    distance = {name: float('inf') for name in map}
    distance[start_station_name] = 0
    
    count=0
    start_time=time.time()
    for _ in range(len(map) - 1):
        for current_name in map:
            count+=1
            linked = map[current_name].links
            for station in linked:
                new_distance = distance[current_name] + geodesic(map[current_name].position, station.position).kilometers
                if new_distance < distance[station.name]:
                    distance[station.name] = new_distance
    end_time=time.time()
    total_time=end_time-start_time
    path = []
    current_name = end_station_name
    while current_name != start_station_name:
        path.insert(0, current_name)
        for station in map[current_name].links:
            if distance[current_name] == distance[station.name] + geodesic(map[station.name].position, map[current_name].position).kilometers:
                current_name = station.name
                break
    path.insert(0, start_station_name)
    return total_time, count, distance[end_station_name], path

def Great_Circle_Distance(position1,position2):
    return geodesic(position1,position2).kilometers

def Manhattan_Distance(position1,position2):
    dist1=geodesic(position1,(position1[0],position2[1])).kilometers
    dist2=geodesic(position2,(position1[0],position2[1])).kilometers
    return dist1+dist2

def Euclidean_Distance(position1,position2):
    dist1=geodesic(position1,(position1[0],position2[1])).kilometers
    dist2=geodesic(position2,(position1[0],position2[1])).kilometers
    return (dist1**2+dist2**2)**0.5

def Chebyshev_Distance(position1,position2):
    dist1=geodesic(position1,(position1[0],position2[1])).kilometers
    dist2=geodesic(position2,(position1[0],position2[1])).kilometers
    return max(dist1,dist2)

def get_path_Astar(start_station_name: str, end_station_name: str, map: dict[str, Station],heuristic) -> tuple[float, int, float,List[str]]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    cost_so_far=dict()
    cost_so_far[start_station_name]=0

    queue = PriorityQueue()
    queue.put((0,[start_station_name]))
    
    count = 0
    start_time=time.time()
    while not queue.empty():
        cost, lst = queue.get()
        current_name=lst[len(lst)-1]
        count+=1
        if end_station_name not in lst:
            linked = map[current_name].links
            for station in linked:
                new_cost=cost_so_far[current_name]+geodesic(map[current_name].position,station.position).kilometers
                if station.name not in cost_so_far or new_cost<cost_so_far[station.name]:
                    cost_so_far[station.name]=new_cost
                    f=new_cost+heuristic(station.position,end_station.position)
                    path=lst.copy()
                    path.append(station.name)
                    queue.put((f,path))
        else:
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost, lst
    return None

def get_path_biAstar(start_station_name:str, end_station_name:str, map:dict[str,Station],heuristic)->tuple[float, int, float,List[str]]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    cost_so_far1=dict()
    cost_so_far2=dict()
    cost_so_far1[start_station_name]=0
    cost_so_far2[end_station_name]=0

    queue1 = PriorityQueue()
    queue2 = PriorityQueue()
    queue1.put((0,[start_station_name]))
    queue2.put((0,[end_station_name]))
    
    count=0
    start_time=time.time()
    while (not queue1.empty()) and (not queue2.empty()):
        cost1, lst1 = queue1.get()
        current_name1=lst1[len(lst1)-1]
        cost2, lst2 = queue2.get()
        current_name2=lst2[len(lst2)-1]
        
        count+=1
        
        if current_name1==current_name2:
            lst2=lst2[:-1][::-1]
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost_so_far1[current_name1]+cost_so_far2[current_name2], lst1+lst2
        if current_name1 in lst2:
            index=lst2.index(current_name1)
            lst2=lst2[:index][::-1]
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost_so_far1[current_name1]+cost_so_far2[current_name2], lst1+lst2
        if current_name2 in lst1:
            index=lst1.index(current_name2)
            lst1=lst1[:index]
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost_so_far1[current_name1]+cost_so_far2[current_name2], lst1+lst2[::-1]
        
        if end_station_name not in lst1:
            linked = map[current_name1].links
            for station in linked:
                new_cost=cost_so_far1[current_name1]+geodesic(map[current_name1].position,station.position).kilometers
                if station.name not in cost_so_far1 or new_cost<cost_so_far1[station.name]:
                    cost_so_far1[station.name]=new_cost
                    f=new_cost+heuristic(station.position,end_station.position)
                    path=lst1.copy()
                    path.append(station.name)
                    queue1.put((f,path))
        else:
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost1, lst1
        
        if start_station_name not in lst2:
            linked = map[current_name2].links
            for station in linked:
                new_cost=cost_so_far2[current_name2]+geodesic(map[current_name2].position,station.position).kilometers
                if station.name not in cost_so_far2 or new_cost<cost_so_far2[station.name]:
                    cost_so_far2[station.name]=new_cost
                    f=new_cost+heuristic(station.position,start_station.position)
                    path=lst2.copy()
                    path.append(station.name)
                    queue2.put((f,path))
        else:
            end_time=time.time()
            total_time=end_time-start_time
            return total_time, count, cost2, lst2
    return None

if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('method',type=str,help='method')
    parser.add_argument('--heuristic', type=str, help='heuristic function', default='Great_Circle')
    
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    method = args.method
    
    stations, underground_lines = build_data()
    
    path=None
    cost=0
    iters=0
    total_time=0
    methodselect = 'get_path_' + method
    method_function = globals().get(methodselect, None)
    
    if(method=='Astar' or method=='biAstar'):
        heuristic=args.heuristic+"_Distance"
        heuristic_function=globals().get(heuristic,None)
        for _ in range(1):
            start_time = time.time()
            _, iters, cost, path=method_function(start_station_name,end_station_name,stations,heuristic_function)
            end_time = time.time()
            once = end_time - start_time
            total_time+=once
        plot_path(path,'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
    else:
        if method_function is not None and callable(method_function):
            for _ in range(1):
                start_time = time.time()
                _, iters, cost, path = method_function(start_station_name, end_station_name, stations)
                end_time = time.time()
                once = end_time - start_time
                total_time+=once
            plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
        else:
            print(f"Method {method} not found or not callable.")
            
    print(f'{method} time consuming: {total_time} s')
    print(f'Iteration count: {iters}')
    print(f'path length: {cost}')