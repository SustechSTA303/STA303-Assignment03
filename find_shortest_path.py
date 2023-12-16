import time
from queue import PriorityQueue
from math import radians, sin, cos, sqrt, atan2
from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
from geopy.distance import geodesic
import argparse
import heapq

def get_path_UCS(start_station_name: str, end_station_name: str, map: dict[str, Station]):
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]

    start_time = time.time()
    interation_times = 0
    
    queue = PriorityQueue()
    queue.put((0, [start_station.name]))
    
    
    while not queue.empty():
        cost, current_path = queue.get()
        current_location = current_path[-1]
        

        if current_location == end_station_name:
            end_time = time.time()
            execution_time = (end_time - start_time)*1000

            return (current_path, cost, interation_times, execution_time)
        
        current_location = map[current_location]
        
        for neighbor in current_location.links:
            interation_times = interation_times + 1
            total_cost = cost + Real(current_location.position, neighbor.position)
            new_path = current_path + [neighbor.name]
            queue.put((total_cost, new_path))
    return None

def get_path_Dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station]):
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    start_time = time.time()
    interation_times = 0
    
    distance = {start_station: 0, end_station: float('inf')}
    visited = set()

    while not (end_station in visited):
        current_station = min((station for station in distance if station not in visited), key=lambda x: distance[x])
        visited.add(current_station)

        for neighbor in current_station.links:
            if neighbor not in visited:
                interation_times = interation_times + 1
                new_distance = distance[current_station] + Real(current_station.position,neighbor.position)
                if neighbor not in distance:
                    distance[neighbor]  = new_distance
                elif new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
    
    path = []
    current = end_station
    while current != start_station:
        path.append(current.name)
        current = min(current.links, key=lambda x: distance[x])

    path.append(start_station.name)
    
    end_time = time.time()
    execution_time = (end_time - start_time)*1000
    
    return (path[::-1],distance[end_station],interation_times,execution_time)

def get_path_Astar(start_station_name: str, end_station_name: str, map: dict[str, Station], heuristic):
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    start_time = time.time()
    interation_times = 0
    
    open_set = PriorityQueue()
    open_set.put((0, [start_station.name]))

    g_score = {}
    g_score[start_station.name] = 0
    
    while not open_set.empty():
        _, current_path = open_set.get()
        current_station = current_path[-1]
        
        if current_station == end_station_name: 
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            
            return (current_path, g_score[current_station], interation_times, execution_time)
        
        current_station = map[current_station]

        for neighbor in current_station.links:
            tentative_g_score = g_score[current_station.name] + Real(current_station.position, neighbor.position)
            if neighbor.name not in g_score or tentative_g_score < g_score[neighbor.name]:
                interation_times = interation_times + 1
                new_path = current_path + [neighbor.name]
                g_score[neighbor.name] = tentative_g_score
                f_score = g_score[neighbor.name] + heuristic(neighbor.position, end_station.position)
                open_set.put((f_score, new_path))
    return []  

def Real(position1,position2):
    return geodesic(position1, position2).kilometers

def Euclidean(position1, position2):
    return (geodesic((position1[0],position1[1]),(position1[0],position2[1])).kilometers ** 2 +
            geodesic((position1[0],position2[1]),(position2[0],position2[1])).kilometers ** 2) ** 0.5
    
def Manhattan(position1, position2):
    return (geodesic((position1[0],position1[1]),(position1[0],position2[1])).kilometers + geodesic((position1[0],position2[1]),(position2[0],position2[1])).kilometers)

def Chebyshev(position1,position2):
    return max(geodesic((position1[0],position1[1]),(position2[0],position1[1])).kilometers,geodesic((position1[0],position2[1]),(position2[0],position2[1])).kilometers,geodesic((position1[0],position1[1]),(position1[0],position2[1])).kilometers,geodesic((position2[0],position1[1]),(position2[0],position2[1])).kilometers)

def Minkowski(position1, position2):
    return (geodesic((position1[0],position1[1]),(position1[0],position2[1])).kilometers ** 3 +
            geodesic((position1[0],position2[1]),(position2[0],position2[1])).kilometers ** 3) ** (1/3)

def get_path_Greedy_BFS(start_station_name: str, end_station_name: str, map: dict[str, Station]):
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    start_time = time.time()
    interation_times = 0
    
    open_set = PriorityQueue()
    open_set.put((0, start_station.name))
    came_from = {}
    came_from[start_station_name] = (None,0)

    while not open_set.empty():
        _, current_station = open_set.get()
        
        if current_station == end_station_name: 
            path = [current_station]
            cost = came_from[current_station][1]
            while current_station != start_station_name:
                current_station = came_from[current_station][0]
                path.insert(0, current_station)
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            
            return (path, cost, interation_times, execution_time)
        
        current_station = map[current_station]

        for neighbor in current_station.links:
            if neighbor.name not in came_from:
                interation_times = interation_times + 1
                f_score = Real(neighbor.position, end_station.position)
                open_set.put((f_score, neighbor.name))
                came_from[neighbor.name] = (current_station.name, came_from[current_station.name][1]+Real(neighbor.position, current_station.position))
    return [] 

    
def get_path_Beam_Search(start_station_name: str, end_station_name: str, map: dict[str, Station], limit):
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    start_time = time.time()
    interation_times = 0
    
    open_set = [(0, [start_station.name])]
    heapq.heapify(open_set)
    g_score = {}
    g_score[start_station.name] = 0
    while open_set:
        current_path = heapq.heappop(open_set)[1]
        current_station = current_path[-1]
        
        if current_station == end_station_name: 
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            
            return (current_path, g_score[current_station], interation_times, execution_time)
        
        current_station = map[current_station]

        for neighbor in current_station.links:
            tentative_g_score = g_score[current_station.name] + Real(current_station.position, neighbor.position)
            if neighbor.name not in g_score or tentative_g_score < g_score[neighbor.name]:
                interation_times = interation_times + 1
                new_path = current_path + [neighbor.name]
                g_score[neighbor.name] = tentative_g_score
                f_score = g_score[neighbor.name] + Real(neighbor.position, end_station.position)
                open_set.extend([(f_score, new_path)])
                open_set = heapq.nsmallest(limit,open_set)
                
    print('fuck')
    return []

def get_path_Biastar(start_station_name: str, end_station_name: str, map: dict[str, Station]) -> List[str]:
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    start_time = time.time()
    interation_times = 0
    
    forward_open_set = PriorityQueue()
    backward_open_set = PriorityQueue()
    
    forward_open_set.put((Real(start_station.position, end_station.position), [start_station_name]))
    backward_open_set.put((Real(start_station.position, end_station.position), [end_station_name]))
    
    forward_g_score = {start_station_name: (0,[start_station_name])}
    backward_g_score = {end_station_name: (0,[end_station_name])}
    
    while not forward_open_set.empty() and not backward_open_set.empty():

        _, forward_current_path = forward_open_set.get()
        forward_current_station = forward_current_path[-1]
        
        if forward_current_station in backward_g_score:
            common_station = forward_current_station
            forward_path = forward_current_path
            backward_path = reversed(backward_g_score[common_station][1][:-1])
            path = forward_path + list(backward_path)
            
            end_time = time.time()
            execution_time = (end_time - start_time)*1000

            return (path, forward_g_score[common_station][0]+backward_g_score[common_station][0], interation_times, execution_time)
        
        forward_current_station = map[forward_current_station]
        

        for neighbor in forward_current_station.links:
            tentative_g_score = forward_g_score[forward_current_station.name][0] + Real(forward_current_station.position, neighbor.position)
            
            if neighbor.name not in forward_g_score or tentative_g_score < forward_g_score[neighbor.name][0]:
                interation_times = interation_times + 1
                new_path = forward_current_path + [neighbor.name]
                forward_g_score[neighbor.name] = (tentative_g_score, new_path)
                forward_f_score = tentative_g_score + Real(neighbor.position, end_station.position)
                forward_open_set.put((forward_f_score, new_path))
                
        _, backward_current_path = backward_open_set.get()
        backward_current_station = backward_current_path[-1]
        
        if backward_current_station in forward_g_score:
            common_station = backward_current_station
            forward_path = forward_g_score[common_station][1][:-1]
            backward_path = reversed(backward_current_path)
            
            path = forward_path + list(backward_path)
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            
            return (path, forward_g_score[common_station][0]+backward_g_score[common_station][0], interation_times, execution_time)
        
        backward_current_station = map[backward_current_station]

        for neighbor in backward_current_station.links:
            tentative_g_score = backward_g_score[backward_current_station.name][0] + Real(backward_current_station.position, neighbor.position)
            if neighbor.name not in backward_g_score or tentative_g_score < backward_g_score[neighbor.name][0]:
                interation_times = interation_times + 1
                new_path = backward_current_path + [neighbor.name]
                backward_g_score[neighbor.name] = (tentative_g_score, new_path)
                backward_f_score = tentative_g_score + Real(neighbor.position, start_station.position)
                backward_open_set.put((backward_f_score, new_path))
                
    return []

def get_path_Weighted_astar(start_station_name: str, end_station_name: str, map: dict[str, Station], weight):
    # You can obtain the Station objects of the starting and ending station through the following code
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    start_time = time.time()
    interation_times = 0
    
    open_set = PriorityQueue()
    open_set.put((0, [start_station.name]))

    g_score = {}
    g_score[start_station.name] = 0
    
    while not open_set.empty():
        _, current_path = open_set.get()
        current_station = current_path[-1]
        
        if current_station == end_station_name: 
            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            
            return (current_path, g_score[current_station], interation_times, execution_time)
        
        current_station = map[current_station]

        for neighbor in current_station.links:
            tentative_g_score = g_score[current_station.name] + Real(current_station.position, neighbor.position)
            if neighbor.name not in g_score or tentative_g_score < g_score[neighbor.name]:
                interation_times = interation_times + 1
                new_path = current_path + [neighbor.name]
                g_score[neighbor.name] = tentative_g_score
                f_score = g_score[neighbor.name] + weight*Real(neighbor.position, end_station.position)
                open_set.put((f_score, new_path))
    return [] 


                

if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    parser.add_argument('limit', type=int, help='limit for beam_search')
    parser.add_argument('weight', type=float, help='weight for weighted_astar')
    parser.add_argument('--algorithm', type=str, help='algrithm', default='Astar')
    parser.add_argument('--heuristic', type=str, help='heristic function', default='Real')
    #eg： python find_shortest_path.py  "Acton Town"  "Stratford" 5 1.2 --algorithm "Astar" --heuristic "Real"
    
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name
    limit = args.limit
    weight = args.weight
    algorithm = args.algorithm
    heuristic = args.heuristic
    heuristic = globals().get(heuristic,None)
    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    
    if algorithm == 'UCS':
        path = get_path_UCS(start_station_name, end_station_name, stations)[0]
    elif algorithm == 'Dijkstra':
        path = get_path_Dijkstra(start_station_name, end_station_name, stations)[0]
    elif algorithm == 'Astar':
        path = get_path_Astar(start_station_name, end_station_name, stations, heuristic)[0]
    elif algorithm == 'Greedy_BFS':
        path = get_path_Greedy_BFS(start_station_name, end_station_name, stations)[0]
    elif algorithm == 'Beam_Search':
        path = get_path_Beam_Search(start_station_name, end_station_name, stations, limit)[0]
    elif algorithm == 'Biastar':
        path = get_path_Biastar(start_station_name, end_station_name, stations)[0]
    elif algorithm == 'Weighted_astar':
        path = get_path_Weighted_astar(start_station_name, end_station_name, stations, weight)[0]
        
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)