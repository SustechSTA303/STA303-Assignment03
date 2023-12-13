import heapq
import math
from geopy.distance import geodesic
from math import radians,cos,sin,acos,asin,sqrt

def Dijkstra(start_station_name, end_station_name, map):
    len_type = 'euclid'
    start_station = map[start_station_name]
    path = {}
    path_list = []
    open_priority = []
    open_dict = {}
    close_table = set()

    open_dict[start_station_name] = 0
    start_station_dict = mkdict(0, start_station)

    heapq.heappush(open_priority, (start_station_dict['priority'], start_station_name, start_station_dict))
    
    while True:
        if len(open_priority) == 0:
            print("open priority is empty!")
            break
        current_node_dict = heapq.heappop(open_priority)[2]
        current_node = current_node_dict['station']
        if current_node.name in close_table:
            continue
        current_prior = open_dict[current_node.name]
        del open_dict[current_node.name]
        close_table.add(current_node.name)
        if current_node.name == end_station_name:
            break
        for neighbor in current_node.links:
            if (neighbor.name not in close_table):
                if (neighbor.name in open_dict):
                    ori_prior = open_dict[neighbor.name]
                    new_prior = current_prior + distance(current_node, neighbor, len_type)
                    if (new_prior < ori_prior):
                        path[neighbor.name] = current_node.name
                        neighbor_dict = mkdict(new_prior, neighbor)
                        heapq.heappush(open_priority, (neighbor_dict['priority'], neighbor.name, neighbor_dict))
                        open_dict[neighbor.name] = new_prior
                else:
                    prior = current_prior + distance(current_node, neighbor, len_type)
                    neighbor_dict = mkdict(prior, neighbor)
                    heapq.heappush(open_priority, (neighbor_dict['priority'], neighbor.name, neighbor_dict))
                    open_dict[neighbor.name] = prior
                    path[neighbor.name] = current_node.name
    
    temp_name = ''
    while True:
        if len(path_list) == 0:
            path_list.append(end_station_name)
            temp_name = end_station_name
        
        path_list.append(path[temp_name])
        temp_name = path[temp_name]

        if temp_name == start_station_name:
            break
    
    return path_list

def mkdict(priority, current_node):
    dic = {}
    dic['station'] = current_node
    dic['priority'] = priority
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
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance
    
