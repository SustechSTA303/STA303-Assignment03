#!/bin/sh

algorithm=(astar greedy_bfs dijikstra)
heuristic=(manhattan euclidean diagonal)

python find_shortest_path.py "South Kensington" "Caledonian Road" "astar" "manhattan"