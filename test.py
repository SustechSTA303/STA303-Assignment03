from build_data import build_data
from find_shortest_path import astar_get_path
from find_shortest_path import bfs_get_path
from find_shortest_path import dfs_get_path
from find_shortest_path import dijkstra_get_path
import time
import pandas as pd

stations, underground_lines = build_data()


def output_with_time(start: str, end: str, map, get_path_func):
    start_time = time.time()
    path = get_path_func(start, end, map)
    end_time = time.time()

    print(f"{get_path_func.__name__}\t: path length = {len(path)}\t, time cost = {end_time - start_time} seconds")


# start_station_name = "Chesham"
# end_station_name = "Upminster" #37
# end_station_name = "Kenton" #21    King's Cross St. Pancras  Stockwell
# end_station_name = "Turnham Green" #2   Ravenscourt Park  Hainault  Rayners Lane   Baker Street   Stratford


station = pd.read_csv('./london/underground_stations.csv')
station_name = station['name']

start_station_name = ''
end_station_name = ''
m = 0
for i in station_name:
    for j in station_name:
        if i != j:
            path = astar_get_path(i, j, stations)
            if len(path) > m:
                start_station_name = i
                end_station_name = j
                m = len(path)

print(f'start station: {start_station_name}, end station: {end_station_name}')
output_with_time(start_station_name, end_station_name, stations, astar_get_path)
output_with_time(start_station_name, end_station_name, stations, bfs_get_path)
output_with_time(start_station_name, end_station_name, stations, dfs_get_path)
output_with_time(start_station_name, end_station_name, stations, dijkstra_get_path)
