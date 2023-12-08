#!/bin/sh

algorithm=(astar greedy_bfs dijikstra)
heuristic=(manhattan euclidean geodesic_distance)

python find_shortest_path.py "South Kensington" "Caledonian Road" "astar" "manhattan"