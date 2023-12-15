import argparse
import time
import numpy as np

from algorithms.build_algorithms import build_algorithms
from build_data import build_data
from evaluation.utils import distance, read_test
from logs.logger import logger


algorithms = ["dijkstra", "bellman ford", "deep first search", "best first search", "A star"]
heuristics = ["euclidean", "manhattan", "diagonal"]

for algorithm in algorithms:
    if algorithm not in ["best first search", "A star"]:
        heuristic = "None"
        log = logger(algorithm, heuristic)

        stations, underground_lines = build_data()
        for i in range(10):
            start_time = time.time()
            file_name = f"evaluation/tests/test_{i}.txt"
            test = read_test(file_name)
            distances = []
            for start_station_name, end_station_name in test:
                path = build_algorithms(algorithm, start_station_name, end_station_name, stations, heuristic)
                dist = distance(path, stations)
                distances.append(dist)
            end_time = time.time()
            run_time = end_time - start_time
            log.update(round(np.mean(distances), 4), round(np.mean(run_time), 4))
        log.log()
        print("Complete")
    else:
        for heuristic in heuristics:
            log = logger(algorithm, heuristic)

            stations, underground_lines = build_data()
            for i in range(10):
                start_time = time.time()
                file_name = f"evaluation/tests/test_{i}.txt"
                test = read_test(file_name)
                distances = []
                for start_station_name, end_station_name in test:
                    path = build_algorithms(algorithm, start_station_name, end_station_name, stations, heuristic)
                    dist = distance(path, stations)
                    distances.append(dist)
                end_time = time.time()
                run_time = end_time - start_time
                log.update(round(np.mean(distances), 4), round(np.mean(run_time), 4))
            log.log()
            print("Complete")