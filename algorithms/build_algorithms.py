from typing import List
from build_data import Station, Node

from .DeepFirstSearch import dfs
from .BellmanFord import bellman_ford
from .Dijkstra import dijkstra
from .BestFirstSearch import bfs
from .Astar import Astar


def build_algorithms(algorithm: str,
                     start_station_name: str,
                     end_station_name: str,
                     map: dict[str, Station],
                     heuristic_function: str = None) -> List[str]:
    if algorithm == "dijkstra":
        return dijkstra(start_station_name, end_station_name, map)

    elif algorithm == "bellman ford":
        return bellman_ford(start_station_name, end_station_name, map)

    elif algorithm == "deep first search":
        return dfs(start_station_name, end_station_name, map)

    elif algorithm == "best first search":
        return bfs(start_station_name, end_station_name, map, heuristic_function)

    elif algorithm == "A star":
        return Astar(start_station_name, end_station_name, map, heuristic_function)

    else:
        raise NotImplementedError(f"Unknown algorithm {algorithm}!")
