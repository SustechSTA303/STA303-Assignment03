from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import math
import time



# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station],search: str) -> (List[str], float):
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
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
#     print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
#     print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    
    if search == 'DFS':
        return DFS(start_station,end_station)
#     elif search == 'DynamicProgramming':
#         return DynamicProgramming(start_station,end_station,start_station)
    
    #The final route
    station = []
    #The station that will be visited
    open_station = [start_station]
    
    #The lat,lon of end_station
    end_position = end_station.position 
    
    
    while open_station:
        #Finding the subway station with the smallest evaluation function
        f = 10.
        min_station = None
        for s in open_station:
            # s.f is evaluation function 
            if s.f < f:
                f = s.f
                min_station = s
        open_station.remove(min_station)
        min_station.visited = True
        
        local_position = min_station.position
        for s in min_station.links:
            cost_new = min_station.cost + cost(local_position,s.position,search)
            
            if s.visited:
                continue
            elif s in open_station:
                if cost_new >= s.cost:
                    continue
            else:
                # s.h is the heuristic function
                s.h = heuristic(s.position,end_position,search)

            s.cost = cost_new
            s.f = s.cost + s.h
            s.parent = min_station
            open_station.append(s)
        
        if end_station in open_station:
            break
           
    route_cost = 0. 
    hascost = end_station.cost != 0.
        
    if end_station.parent:
        s = end_station
        station.append(s.name)
        while s.parent:
            if not hascost:
                route_cost += Euclidean(s.position,s.parent.position)
            s = s.parent
            station.append(s.name)
    
    if hascost:
        route_cost = end_station.cost
    
    station.reverse()
    return station,route_cost

def cost(pos1,pos2,search):
    result = 0.
    if search != 'Best-First-Search':
        result = Euclidean(pos1,pos2)
    return result

def heuristic(pos1,pos2,search):
    result = 0.
    if search == 'Dijkstra':
        result = 0.
    elif search == 'A* Euclidean distance':
        result = Euclidean(pos1,pos2)
    elif search == 'A* Manhattan distance':
        result = Manhattan(pos1,pos2)
    elif search == 'A* Octile distance':
        result = Octile(pos1,pos2)
    elif search == 'Best-First-Search':
        result = Euclidean(pos1,pos2)
    elif search == 'A* Chebyshev distance':
        result = Chebyshev(pos1,pos2)
    return result
    
    
def Euclidean(pos1,pos2):
    x1,y1 = pos1
    x2,y2 = pos2
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def Manhattan(pos1,pos2):
    x1,y1 = pos1
    x2,y2 = pos2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return dx+dy

def Octile(pos1,pos2):
    x1,y1 = pos1
    x2,y2 = pos2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    k = math.sqrt(2) - 1
    return max(dx, dy) + k * min(dx, dy)

def Chebyshev(pos1,pos2):
    x1,y1 = pos1
    x2,y2 = pos2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx,dy)

def DFS(start_station, end_station):
    stack = [start_station]
    while stack:
        station = stack[len(stack)-1]
        stack.remove(station)
        station.visited = True
        for s in station.links:
            if s.visited:
                continue
            s.visited = True
            stack.append(s)
            s.parent = station
            if(s == end_station):
                route_cost = 0.
                route = []            
                route.append(s.name)
                while s.parent: 
                    route_cost += Euclidean(s.position,s.parent.position)
                    s = s.parent
                    route.append(s.name)
                return route, route_cost
                
# def DynamicProgramming(station, end_station, start_station):
#     if station == end_station:
#         return 0.
    
#     mincost = 10.
#     minstation = None
#     for s in station.links:
#         if s.visited:   
#             continue
#         s.visited = True
#         _,dpcost = DynamicProgramming(s,end_station,start_station)
#         localcost = Euclidean(station.position,s.position)+dpcost
            
#         if localcost < mincost:
#             mincost = localcost
#             minstation = s
#     s.parent = start_station
    
#     if station == start_station:
#         route_cost = 0.
#         route = []         
#         s = end_station
#         route.append(s.name)
#         while s.parent: 
#             route_cost += Euclidean(s.position,s.parent.position)
#             s = s.parent
#             route.append(s.name)    
#         return route, route_cost
    
#     return [],mincost 
    
if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('search', type=str, help='search')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    search = args.search

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path,route_cost = get_path(start_station_name, end_station_name, stations, search)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
#     plot_path(path, f'visualization_underground/{search}_my_shortest_path_in_London_railway.html', stations, underground_lines)
    print(search)
    
# visualization the path
# Open the visualization_underground/my_all_path_in_London_railway.html to view all path, and your paths is marked in different colors.
def plot_all_path(paths, searchs):
    plot_path(paths, f'visualization_underground/my_all_path_in_London_railway.html', stations, underground_lines, searchs)
    

      
    