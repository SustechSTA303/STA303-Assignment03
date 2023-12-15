from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
import queue
import time
import matplotlib.pyplot as plt
import numpy as np

file_path = "output/output.txt"

#draw results as histograms
def draw_details(time,length,start,end):
    data = {
    "Time_cost": time,
    "Path_length": length
   }
    type_ = ['BFS','Dijkstra','Manh1','Manh10','Manh100','Manh1000','Euc1','Euc10','Euc100','Euc1000','Oct1','Oct10','Oct100','Oct1000']
   
    fig, ax = plt.subplots(figsize=(8,8))
    bar_width = 0.45
    index = np.arange(len(data["Time_cost"]))
    bar1 = ax.bar(index, data["Time_cost"], bar_width, label='Time_cost')

    
    cnt = 0
    for bar in bar1 :
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025, f'len:{length[cnt]}' , ha='center', va='baseline')
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}ms', ha='center', va='bottom')
        cnt+=1
    
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Time_costs')
    ax.set_title('Time costs of each algorithm and respective path length')
    ax.set_xticks(index + bar_width / 2,type_, rotation=45, ha='right', fontsize=10)
    plt.tight_layout() 

    plt.savefig('output/chart' + f'_{start}&{end}' + '.png')

def struct_path(prev,current):
    if isinstance(prev,dict) and isinstance(current,Station):
        path = []
        while prev[current] is not None:
            path.append(current.name)
            current = prev[current]
        path.append(start_station_name)
        return path

# Euclid distance
def heuristic_Euclid(a,b):
    if isinstance(a,Station) and isinstance(b,Station):
        lat_a,lon_a = a.position
        lat_b,lon_b = b.position
        return pow(((lat_a - lat_b)**2 + (lon_a - lon_b)**2),0.5 )

#Manhattan distance
def heuristic_Manhattan(a,b):
    if isinstance(a,Station) and isinstance(b,Station):
        lat_a,lon_a = a.position
        lat_b,lon_b = b.position
        return abs(lat_a - lat_b) + abs(lon_a - lon_b)
    
#This heuristic function allow moving at eight direction
def heuristic_octCardinal(a,b):
     if isinstance(a,Station) and isinstance(b,Station):
        lat_a,lon_a = a.position
        lat_b,lon_b = b.position
        dlat = abs(lat_a - lat_b)
        dlon = abs(lon_a - lon_b)
        return abs(dlat - dlon) + 1.414 * min(dlat,dlon)
        
# cost between a pair of stations
def distance(a,b):
    if isinstance(a,Station) and isinstance(b,Station):
        return 1
    
def BFS_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]):
    """
    runs BFS on the map, find the shortest path between a and b
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
    prev = {}
    prev[start_station] = None
    Queue_ = queue.Queue()
    start_time = time.process_time_ns()
    Queue_.put(start_station)
    while Queue_:
        current = Queue_.get()
        if current.name == end_station_name:
            end_time = time.process_time_ns()
            exe_time = (end_time - start_time) / 1000000
            return struct_path(prev,current),round(exe_time,3)
        for link in current.links:
            if link not in prev:
                Queue_.put(link)
                prev[link] = current
                
            
            
def Dijkstra_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    """
    runs dijkstra on the map, find the shortest path between a and b
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
    #start_station = map[start_station_name]
    #end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    
    cost = {}
    prev = {}
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    prev[start_station] = None
    PQ = []
    openset = []
    openset.append(start_station)
    for station in map.values():
        cost[station] = float("inf")
        prev[station] = None
        
    cost[start_station] = 0    
    start_time = time.process_time_ns()
    heapq.heappush(PQ,(cost[start_station],start_station))    
    while PQ:
        _,current = heapq.heappop(PQ)
        #openset.remove(current)
        if current.name == end_station_name:
            end_time = time.process_time_ns()
            exe_time = (end_time - start_time) / 1000000
            return struct_path(prev,current),round(exe_time,3)
        for link in current.links:
            tentative_cost = cost[current] + distance(current,link)
            if tentative_cost < cost[link]:
                cost[link] = tentative_cost
                prev[link] = current
                heapq.heappush(PQ,(cost[link],link)) 

    
    

def A_star_get_path(start_station_name: str, end_station_name: str, map: dict[str, Station],heuristic_type: str,heuristic_scale: float):
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
    fScore_heap = []
    prev = {}
    gScore = {}
    fScore = {}
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    prev[start_station] = None
    for station in map.values():
        gScore[station] = float("inf")
        fScore[station] = float("inf")
        if station.name == start_station.name:
            gScore[station] = 0
            if heuristic_type == 'Manh':
                 fScore[station] = heuristic_scale * heuristic_Manhattan(start_station,end_station)
            elif heuristic_type == 'Euc':
                 fScore[station] = heuristic_scale * heuristic_Euclid(start_station,end_station)
            elif heuristic_type == 'oct':
                 fScore[station] = heuristic_scale * heuristic_octCardinal(start_station,end_station)
        
    start_time = time.process_time_ns()
    heapq.heappush(fScore_heap,(fScore[start_station],start_station))
    while fScore_heap:
        _,current = heapq.heappop(fScore_heap)
        if current.name == end_station.name:
            end_time = time.process_time_ns()
            exe_time = (end_time - start_time) / 1000000
            return struct_path(prev,current),round(exe_time,3)
        
        for link in current.links:
            tentative_gScore = gScore[current] + distance(current,link)
            if tentative_gScore < gScore[link]:
                prev[link] = current
                gScore[link] = tentative_gScore 
                if heuristic_type == 'Manh':
                    fScore[link] = (tentative_gScore + heuristic_scale * heuristic_Manhattan(link,end_station))
                elif heuristic_type == 'Euc':
                    fScore[link] = (tentative_gScore + heuristic_scale * heuristic_Euclid(link,end_station))
                elif heuristic_type == 'oct':
                    fScore[link] = (tentative_gScore + heuristic_scale * heuristic_octCardinal(link,end_station))
                heapq.heappush(fScore_heap,(fScore[link],link))
                    

def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]):
    
    path,time = A_star_get_path(start_station_name,end_station_name,map,'Manh',1)
    print(f'Stations of path that A_star found is {len(path)},cost {time} ms, heuristic function is Manhattan')
    
    return path



def evaluation(start_station_name: str, end_station_name: str, map: dict[str, Station]):
    times = []
    length = []
    with open(file_path, 'a') as file:
        file.write("\n" + "\n" + f'Test of {start_station_name} and {end_station_name} begin:' + "\n")
    path1,time1 = BFS_get_path(start_station_name,end_station_name,map)
    path2,time2 = Dijkstra_get_path(start_station_name,end_station_name,map)
    times.append(time1)
    times.append(time2)
    length.append(len(path1))
    length.append(len(path2))
    scale = [1,10,100,1000]
    heus = ['Manh','Euc','oct']
    with open(file_path, 'a') as file:
        result1 = f'BFS, Stations of path {len(path1)}, cost {time1} ms'
        result2 = f'Dijkstra, Stations of path {len(path2)}, cost {time2} ms'
        file.write("\n" + result1 + "\n" + result2)
    for heu in heus:
        for scaling in scale:
            path3,time3 = A_star_get_path(start_station_name,end_station_name,map,heu,scaling)
            times.append(time3)
            length.append(len(path3))
            with open(file_path, 'a') as file:
                result = f'Astar, Stations of path {len(path3)}, cost {time3} ms, heuristic {heu}, scaled by {scaling}'
                file.write("\n" + result)
    return times,length
    
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
    for (start,end) in zip(('Richmond','Eastcote','Canning Town','Wimbledon','Green Park','Mile End','Victoria','Waterloo'),
                           ('Loughton','Greenwich','Kenton','Oakwood','Canary Wharf','Hammersmith','Bank','Paddington')):
        times,length = evaluation(start,end,stations)
        draw_details(times,length,start,end)
        
    path = get_path(start_station_name, end_station_name, stations)
        
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
