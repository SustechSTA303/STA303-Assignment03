from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
import queue
import math


class PriorityQueue(object):  #define a class of priority queue
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return True if not self._queue else False


def cost(current_station:Station,next_station:Station,Cost): # for different selections of movement cost, return corresponding cost value.
    if(Cost=='UNIT'):
        return 1
    if(Cost=='DISTANCE'):
        current_station_lat,current_station_lon=current_station.position
        next_station_lat,next_station_lon=next_station.position
        return math.sqrt((current_station_lat-next_station_lat)**2+(current_station_lon-next_station_lon)**2)

    
def heuristic(station: Station,end_station:Station,Cost):  #Manhatton Distance as heuristic
    station_lat,station_lon=station.position
    end_station_lat,end_station_lon=end_station.position
    distance=abs(station_lat-end_station_lat)+abs(station_lon-end_station_lon)
    if(Cost=='UNIT'):
        return distance*50  #the scaling should be chosen properly. 
    if(Cost=='DISTANCE'):
        return distance/2

def heuristic_l2(station: Station,end_station:Station,Cost):# Euclidean distance as heuristic
    station_lat,station_lon=station.position
    end_station_lat,end_station_lon=end_station.position
    distance= math.sqrt((station_lat-end_station_lat)**2+(station_lon-end_station_lon)**2)
    if(Cost=='UNIT'):
        return distance*50
    if(Cost=='DISTANCE'):
        return distance/2
    
    
    
def get_path_dijkstra(start_station_name: str, end_station_name: str, map: dict[str, Station],Cost):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    pq=PriorityQueue()
    came_from=dict()
    cost_so_far=dict()
    List=[]
    
    cost_so_far[start_station]=0
    came_from[start_station]=None
    
    pq.push(start_station,0)
    
    while not pq.empty():
        current_station=pq.pop()
        if(current_station==end_station):
            break
        for station in current_station.links:
            new_cost=cost_so_far[current_station]+cost(current_station,station,Cost)
            if (station not in cost_so_far or cost_so_far[station]>new_cost):
                cost_so_far[station]=new_cost
                pq.push(station,cost_so_far[station])
                came_from[station]=current_station
    while came_from[current_station] is not None:
        List.append(current_station.name)
        current_station=came_from[current_station]
    List.append(start_station.name)
    List.reverse()
    return List


def get_path_bfs(start_station_name: str, end_station_name: str, map: dict[str, Station],Cost):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    q=queue.Queue()
    came_from=dict()
    List=[]
    reached=set()
    
    q.put(start_station)
    came_from[start_station]=None
    
    reached.add(start_station)
    while not q.empty():
        current_station=q.get()
        if(current_station==end_station):
            break
        for next in current_station.links:
            if next not in reached:
                came_from[next]=current_station
                q.put(next)
                reached.add(next)
    while came_from[current_station] is not None:
        List.append(current_station.name)
        current_station=came_from[current_station]
    List.append(start_station.name)
    List.reverse()
    return List


def get_path_A_Manhatton(start_station_name: str, end_station_name: str, map: dict[str, Station],Cost): #-> List[str]:
    pq=PriorityQueue()
    came_from=dict()
    cost_so_far=dict()
    List=[]
    
    start_station = map[start_station_name]
    end_station = map[end_station_name]
 
    
    cost_so_far[start_station]=0
    came_from[start_station]=None

    pq.push(start_station,heuristic(start_station,end_station,Cost))
    while not pq.empty():
        current_station=pq.pop()
        if(current_station==end_station):
            break
        for station in current_station.links:
            new_cost=cost_so_far[current_station]+cost(current_station,station,Cost)
            if(station not in cost_so_far or cost_so_far[station]>new_cost):
                cost_so_far[station]=new_cost
                pq.push(station,cost_so_far[station]+heuristic(station,end_station,Cost))
                came_from[station]=current_station

    while came_from[current_station] is not None:
        List.append(current_station.name)
        current_station=came_from[current_station]
    List.append(start_station.name)
    List.reverse()
    return List


def get_path_A_L2(start_station_name: str, end_station_name: str, map: dict[str, Station],Cost): #-> List[str]:
    pq=PriorityQueue()
    came_from=dict()
    cost_so_far=dict()
    List=[]
    
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    
    
    cost_so_far[start_station]=0
    came_from[start_station]=None

    pq.push(start_station,heuristic_l2(start_station,end_station,Cost))
    while not pq.empty():
        current_station=pq.pop()
        if(current_station==end_station):
            break
        for station in current_station.links:
            new_cost=cost_so_far[current_station]+cost(current_station,station,Cost)
            if(station not in cost_so_far or cost_so_far[station]>new_cost):
                cost_so_far[station]=new_cost
                pq.push(station,cost_so_far[station]+heuristic_l2(station,end_station,Cost))
                came_from[station]=current_station

    while came_from[current_station] is not None:
        List.append(current_station.name)
        current_station=came_from[current_station]
    List.append(start_station.name)
    List.reverse()
    
    return List
    
    