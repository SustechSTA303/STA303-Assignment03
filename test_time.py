import timeit

from build_data import Station, build_data
from find_shortest_path import get_path
from manhattan_distance import get_path_m
from chebyshev_distance import get_path_c
from d import get_path_dijkstra
from weighted_a_star import weighted_a_star

stations, underground_lines = build_data()

#setA
stations, underground_lines = build_data()
start_station_name1="Acton Town"
end_station_name1="Turnham Green"
#setB
start_station_name3="Acton Town"
end_station_nam3="East Acton"

#setC
start_station_name2="Acton Town"
end_station_name2="King's Cross St. Pancras"

#setD
start_station_name="Acton Town"
end_station_name="Canada Water"

# 测试 A* 算法 欧几里得距离
start_time1= timeit.default_timer()
path,e1, iterations= get_path(start_station_name, end_station_name, stations)
end_time1=timeit.default_timer()
print(f"A* Algorithm  Euclidean distance: {end_time1 - start_time1} seconds")
# 测试 A*算法 曼哈顿距离
start_time2= timeit.default_timer()
path,e2, iteration = get_path_m(start_station_name, end_station_name, stations)
end_time2=timeit.default_timer()
print(f"A* Algorithm  Manhattan distance: {end_time2 - start_time2} seconds")
#A*算法 切比雪夫距离
start_time3= timeit.default_timer()
path,e3, iterations  = get_path_c(start_station_name, end_station_name, stations)
end_time3=timeit.default_timer()
print(f"A* Algorithm  Chebyshev distance: {end_time3 - start_time3} seconds")
#Dijkstras算法
start_time4= timeit.default_timer()
path = get_path_dijkstra(start_station_name, end_station_name, stations)
end_time4=timeit.default_timer()
print(f"Dijkstras: {end_time4 - start_time4} seconds")
#加权a*算法
start_time5= timeit.default_timer()
path = weighted_a_star(start_station_name, end_station_name, stations)
end_time5=timeit.default_timer()
print(f"weighted_a_star: {end_time5 - start_time5} seconds")
