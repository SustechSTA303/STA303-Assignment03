import math
import queue
from math import radians,cos,sin,acos,asin,sqrt

def Astar(start_station_name, end_station_name, map):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    path = {}
    path_list = []
    open_priority = queue.PriorityQueue()
    open_table = set()
    close_table = set()

    start_station_dict = mkdir(0, start_station, start_station, end_station)

    open_priority.put((start_station_dict['priority'], start_station_name, start_station_dict))
    open_table.add(start_station.name)

    break_flag = False
    while not break_flag:
        current_node_dict = open_priority.get()[2]
        current_node = current_node_dict['station']
        current_G = current_node_dict['G']
        open_table.remove(current_node.name)
        if current_node.name in close_table:
            continue
        
        for neighbor_node in current_node.links:
            if (neighbor_node.name not in open_table) and (neighbor_node.name not in close_table):
                neighbor_dict = mkdir(current_G, current_node, neighbor_node, end_station)
                open_priority.put((neighbor_dict['priority'], neighbor_node.name, neighbor_dict))
                open_table.add(neighbor_node.name)
                path[neighbor_node.name] = current_node.name
                if neighbor_node.name == end_station_name:
                    break_flag = True
                    break
         
        close_table.add(current_node.name)

        if open_priority.qsize() == 0:
            print("open priority queue is empty!(end)")
            break

    temp_name = ''
    while True:
        if len(path_list) == 0:
            path_list.append(end_station.name)
            temp_name = end_station.name
        
        path_list.append(path[temp_name])
        temp_name = path[temp_name]

        if temp_name == start_station_name:
            break
    
    return path_list
        
        
def mkdir (last_G, last_station, current_station, end_station):
    len_type = 'geo'
    dic = {}
    dic['station'] = current_station
    dic['G'] = last_G + distance(last_station, current_station, len_type)
    dic['H'] = distance(current_station, end_station, len_type)
    dic['priority'] = dic['G'] + dic['H']
    return dic

def distance(node_1, node_2, len_type):
    node_1_location = (node_1.position[0], node_1.position[1])
    node_2_location = (node_2.position[0], node_2.position[1])
    if len_type == 'manhattan':
        return float(abs(node_1_location[0] - node_2_location[0]) + abs(node_1_location[1] - node_2_location[1]))
    elif len_type == 'euclid':
        return math.sqrt((node_1_location[0] - node_2_location[0])**2 + (node_1_location[1] - node_2_location[1])**2)
    elif len_type == 'geo':
        return geodistance(node_1_location[1], node_1_location[0], node_2_location[1], node_2_location[0])
    
def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    distance=round(distance/1000,3)
    return distance