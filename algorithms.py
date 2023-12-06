import time
from pympler import asizeof
from queue import PriorityQueue
import math

def Haversine_Distance(start_station,end_station):
    lat1 = start_station.position[0]
    lon1 = start_station.position[1]
    lat2 = end_station.position[0]
    lon2 = end_station.position[1]
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c
    return d

def manhattan_Distance(start_station,end_station):
    lat1 = start_station.position[0]
    lon1 = start_station.position[1]
    lat2 = end_station.position[0]
    lon2 = end_station.position[1]
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c
    return d

def Astar(start_station_name,end_station_name,map,distance_func):
    if distance_func == 'manhattan':
        distance_func = manhattan_Distance
    elif distance_func == 'haversine':
        distance_func = Haversine_Distance

    start_station = map[start_station_name]
    end_station = map[end_station_name]
    start_time = time.time()
    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = {}
    cost_so_far = {}
    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0
    while not frontier.empty():
        current = frontier.get()[1]
        if current == end_station:
            break
        for next in current.links:
            new_cost = cost_so_far[current.name] + distance_func(next,current)
            if next.name not in cost_so_far or new_cost < cost_so_far[next.name]:
                cost_so_far[next.name] = new_cost
                priority = new_cost + Haversine_Distance(next,end_station)
                frontier.put((priority, next))
                came_from[next.name] = current.name
    end_time = time.time()
    print('Time:',end_time-start_time)
    total_memory_usage = asizeof.asizeof(map) + asizeof.asizeof(start_station_name) + asizeof.asizeof(end_station_name)
    print('Total memory usage:',total_memory_usage)
    path = []
    current = end_station_name
    while current != start_station_name:
        path.append(current)
        current = came_from[current]
    path.append(start_station_name)
    path.reverse()
    return path


def UCS(start_station_name,end_station_name,map,distance_func):
    if distance_func == 'manhattan':
        distance_func = manhattan_Distance
    elif distance_func == 'haversine':
        distance_func = Haversine_Distance

    start_station = map[start_station_name]
    end_station = map[end_station_name]
    start_time = time.time()
    frontier = PriorityQueue()
    frontier.put((0, start_station))
    came_from = {}
    cost_so_far = {}
    came_from[start_station_name] = None
    cost_so_far[start_station_name] = 0
    while not frontier.empty():
        current = frontier.get()[1]
        if current == end_station:
            break
        for next in current.links:
            new_cost = cost_so_far[current.name] + distance_func(next,current)
            if next.name not in cost_so_far or new_cost < cost_so_far[next.name]:
                cost_so_far[next.name] = new_cost
                priority = new_cost
                frontier.put((priority, next))
                came_from[next.name] = current.name
    end_time = time.time()
    print('Time:',end_time-start_time)
    total_memory_usage = asizeof.asizeof(map) + asizeof.asizeof(start_station_name) + asizeof.asizeof(end_station_name)
    print('Total memory usage:',total_memory_usage)
    path = []
    current = end_station_name
    while current != start_station_name:
        path.append(current)
        current = came_from[current]
    path.append(start_station_name)
    path.reverse()
    return path

def BFS(start_station_name,end_station_name,map,distance_func):

    start_station = map[start_station_name]
    end_station = map[end_station_name]
    start_time = time.time()
    frontier = []
    frontier.append(start_station)
    came_from = {}
    came_from[start_station_name] = None
    while frontier:
        current = frontier.pop(0)
        if current == end_station:
            break
        for next in current.links:
            if next.name not in came_from:
                frontier.append(next)
                came_from[next.name] = current.name
    end_time = time.time()
    total_time = end_time-start_time
    print('Time:',total_time)
    total_memory_usage = asizeof.asizeof(map) + asizeof.asizeof(start_station_name) + asizeof.asizeof(end_station_name)
    print('Total memory usage:',total_memory_usage)
    path = []
    current = end_station_name
    while current != start_station_name:
        path.append(current)
        current = came_from[current]
    path.append(start_station_name)
    path.reverse()
    return path

def cal_total_cost(path,map,dis_func):
    if dis_func == 'manhattan':
        dis_func = manhattan_Distance
    elif dis_func == 'haversine':
        dis_func = Haversine_Distance
    total_cost = 0
    for i in range(len(path)-1):
        total_cost += dis_func(map[path[i]],map[path[i+1]])
    return total_cost
