from queue import Queue

def Bidirectional(start_station_name, end_station_name, map):
    start_station = map[start_station_name]
    end_station = map[end_station_name]
    start_path = {}
    end_path = {}
    path_list = []
    start_queue = Queue()
    end_queue = Queue()
    start_set = set()
    end_set = set()
    start_queue.put(start_station)
    end_queue.put(end_station)
    start_set.add(start_station.name)
    end_set.add(end_station.name)

    while True:
        current_start = start_queue.get()
        current_end = end_queue.get()
        start_set.add(current_start)
        end_set.add(current_end)
        for start_neighbor in current_start.links:
            if start_neighbor.name == end_station_name:
                start_path[end_station_name] = current_start.name
                start_set.add(end_station_name)
                break
            if start_neighbor in start_set:
                continue
            start_queue.put(start_neighbor)
            start_set.add(start_neighbor.name)
            start_path[start_neighbor.name] = current_start.name
        
        for end_neighbor in current_end.links:
            if end_neighbor.name == start_station_name:
                end_path[start_station_name] = current_end.name
                end_set.add(start_station_name)
                break
            if end_neighbor in end_set:
                continue
            end_queue.put(end_neighbor)
            end_set.add(end_neighbor.name)
            end_path[end_neighbor.name] = current_end.name

        if start_set.intersection(end_set):
            break

    start_name = ''
    end_name = ''
    intersection = list(start_set.intersection(end_set))[0]
    while True:
        if len(path_list) == 0:
            path_list.append(intersection)
            start_name = intersection
            end_name = intersection

        if start_name != start_station_name:
            start_name = start_path[start_name]
            path_list.insert(0, start_name)
                
        if end_name != end_station_name:
            end_name = end_path[end_name]
            path_list.append(end_name)

        if (start_name == start_station_name) and (end_name == end_station_name):
            break

    return path_list

                

                
        


        




