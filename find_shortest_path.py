from collections import deque

from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import heapq
import time
import math
def euclidean_distance(start, goal):
    return math.sqrt(((goal.position[0] - start.position[0])*110.574) ** 2 + ((goal.position[1] - start.position[1])*111.32) ** 2)
def manhattan_distance(start, goal):
    return abs(goal.position[0] - start.position[0]) + abs(goal.position[1] - start.position[1])

def dijkstra_num(graph, start, end):
    distances = {vertex: float('infinity') for vertex in graph}
    fars= {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    fars[start] = 0
    previous_vertices = {vertex: None for vertex in graph}
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor in graph[current_vertex].links:
            far = fars[current_vertex] + euclidean_distance(neighbor,graph[current_vertex])
            distance = distances[current_vertex] + 1
            # print(neighbor.name)
            # print(distances.keys())
            # print(distances[neighbor.name])
            if distance < distances[neighbor.name]:
                fars[neighbor.name]=far
                distances[neighbor.name] = distance
                previous_vertices[neighbor.name] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor.name))
    # 路径
    path = []
    current_vertex = end
    while previous_vertices[current_vertex] is not None:
        path.insert(0, current_vertex)
        current_vertex = previous_vertices[current_vertex]
    path.insert(0, start)
    return path,distances[end],fars[end]

def dijkstra_dis(graph, start, end):
    distances = {vertex: float('infinity') for vertex in graph}
    fars= {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    fars[start] = 0
    previous_vertices = {vertex: None for vertex in graph}
    priority_queue = [(0, start)]
    while priority_queue:
        current_far, current_vertex = heapq.heappop(priority_queue)
        if current_far > fars[current_vertex]:
            continue
        for neighbor in graph[current_vertex].links:
            far = fars[current_vertex] + euclidean_distance(neighbor,graph[current_vertex])
            distance = distances[current_vertex] + 1
            # print(neighbor.name)
            # print(distances.keys())
            # print(distances[neighbor.name])
            if far < fars[neighbor.name]:
                fars[neighbor.name]=far
                distances[neighbor.name] = distance
                previous_vertices[neighbor.name] = current_vertex
                heapq.heappush(priority_queue, (far, neighbor.name))
    # 路径
    path = []
    current_vertex = end
    while previous_vertices[current_vertex] is not None:
        path.insert(0, current_vertex)
        current_vertex = previous_vertices[current_vertex]
    path.insert(0, start)
    return path,distances[end],fars[end]

def bfs(graph,start_name, end_name):
    start=graph[start_name]
    end=graph[end_name]
    far=0
    queue = deque([(start, [start.name])])
    visited = set()
    while queue:
        current_station, path = queue.popleft()
        visited.add(current_station)
        if current_station == end:
            return path
        for neighbor in current_station.links:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor.name]))
                visited.add(neighbor)
    return None

def a_star_search(graph, start_name, goal_name, heuristic_function):
    start = graph[start_name]
    goal = graph[goal_name]
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    path = []
    while open_set:
        current_score, current_station = heapq.heappop(open_set)
        if current_station == goal:
            while current_station in came_from:
                path.append(current_station.name)
                current_station = came_from[current_station]
            path.append(start.name)
            return (path[::-1],g_score[graph[end_station_name]])
        for neighbor in current_station.links:
            tentative_g_score = g_score[current_station] + euclidean_distance(neighbor,current_station)
            # tentative_g_score = g_score[current_station] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic_function(graph[neighbor.name], goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current_station
    return None  # 未找到路径

# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station], algorithm):
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
    # start_station = map[start_station_name]
    # end_station = map[end_station_name]

    # Given a Station object, you can obtain the name and latitude and longitude of that Station by the following code
    # print(f'The longitude and latitude of the {start_station.name} is {start_station.position}')
    # print(f'The longitude and latitude of the {end_station.name} is {end_station.position}')
    # for item in start_station.links:
    #     print(item.name)
    total_time = 0
    path=[]
    if(algorithm=="dijkstra_num"):
        for _ in range(50):
            start_time = time.time()
            path,distance,far=dijkstra_num(map, start_station_name, end_station_name)
            end_time = time.time()
            # print(path)
            total_time += (end_time - start_time)
        # print(total_time)
        average_time = total_time* 1000.0 / 50
        print("The time of dijkstra_num algorithm is", float(average_time), "ms")
        print("The path length is",far,"km")
        print("The station number is",len(path)-1,)
        plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway_dijkstra_num.html', stations,underground_lines)
    elif(algorithm=="dijkstra_dis"):
        for _ in range(50):
            start_time = time.time()
            path,distance,far=dijkstra_dis(map, start_station_name, end_station_name)
            end_time = time.time()
            # print(path)
            total_time += (end_time - start_time)
        # print(total_time)
        average_time = total_time* 1000.0 / 50
        print("The time of dijkstra_dis algorithm is", float(average_time), "ms")
        print("The path length is",far,"km")
        print("The station number is",len(path)-1)
        plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway_dijkstra_dis.html', stations,underground_lines)
    elif(algorithm=="a_star_search"):
        for i in (euclidean_distance,manhattan_distance):
            if(i==euclidean_distance):
                name="euclidean_distance"
            elif(i==manhattan_distance):
                name="manhattan_distance"
            # else:
            #     name="diagonal_distance"
            for _ in range(50):
                start_time = time.time()
                path,distance=a_star_search(map, start_station_name, end_station_name,i)
                end_time = time.time()
                # print(path)
                total_time += (end_time - start_time)
            # print(total_time)
            average_time = total_time*1000.0 /50
            print("The time of Astar algorithm with",name,"is",float(average_time), "ms")
            print("The path length is", distance, "km")
            print("The station number is",len(path)-1)
            plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway_'+algorithm+'_'+name+'.html', stations, underground_lines)
    elif(algorithm=="bfs"):
        for _ in range(50):
            start_time = time.time()
            path=bfs(map, start_station_name, end_station_name)
            end_time = time.time()
            # print(path)
            total_time += (end_time - start_time)
        # print(total_time)
        distance=len(path)-1
        far=0
        # print(path)
        for j in range(len(path)-1):
            far += euclidean_distance(map[path[j]], map[path[j + 1]])
        average_time = total_time* 1000.0 / 50
        print("The time of bfs algorithm is", float(average_time), "ms")
        print("The path length is",far,"km")
        print("The station number is",len(path)-1)
        plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway_dijkstra_dis.html', stations,underground_lines)
    pass

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
    for i in ("dijkstra_num","dijkstra_dis","a_star_search","bfs"):
        path=get_path(start_station_name,end_station_name,stations,i)
    # path = get_path("Acton Town", "Turnham Green", stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red



