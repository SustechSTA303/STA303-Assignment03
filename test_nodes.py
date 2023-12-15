import sys

from build_data import Station, build_data
from find_shortest_path import get_path
from manhattan_distance import get_path_m
from chebyshev_distance import get_path_c
from d import get_path_dijkstra
from weighted_a_star import weighted_a_star

#setA
stations, underground_lines = build_data()
start_station_name1="Acton Town"
end_station_name1="Turnham Green"
#setB
start_station_name3="Acton Town"
end_station_name3="East Acton"

#setC
start_station_name2="Acton Town"
end_station_name2="King's Cross St. Pancras"

#setD
start_station_name="Acton Town"
end_station_name="Canada Water"

path1,e1, iterations1 = get_path(start_station_name, end_station_name, stations)
print(f"Euclidean distance Expanded nodes:{e1}")

path,e2,iterations2 = get_path_m(start_station_name, end_station_name, stations)
print(f"Manhattan distance Expanded nodes:{e2}")

path,e3,iterations3 = get_path_c(start_station_name, end_station_name, stations)
print(f"Chebyshev distance Expanded nodes:{e3}")


print(f"Euclidean distance Iterations:{iterations1}")
print(f"Manhattan distance Iterations:{iterations2}")
print(f"Chebyshev distance Iterations:{iterations3}")

