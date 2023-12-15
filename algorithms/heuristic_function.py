from typing import List
import math

from build_data import Station


def build_heuristic_function(heuristic_function):
    if heuristic_function == 'euclidean':
        return euclidean_distance

    elif heuristic_function == 'manhattan':
        return manhattan_distance

    elif heuristic_function == 'diagonal':
        return diagonal_distance

    else:
        raise NotImplementedError('Unknown heuristic function!')


def euclidean_distance(current_station_name: str,
                       end_station_name: str,
                       map: dict[str, Station]) -> float:
    current_station = map[current_station_name]
    end_station = map[end_station_name]
    lat_distance = current_station.position[0] - end_station.position[0]
    lon_distance = current_station.position[1] - end_station.position[1]
    return math.sqrt(lat_distance ** 2 + lon_distance ** 2)


def manhattan_distance(current_station_name: str,
                       end_station_name: str,
                       map: dict[str, Station]) -> float:
    current_station = map[current_station_name]
    end_station = map[end_station_name]
    lat_distance = current_station.position[0] - end_station.position[0]
    lon_distance = current_station.position[1] - end_station.position[1]
    return abs(lat_distance) + abs(lon_distance)


def diagonal_distance(current_station_name: str,
                      end_station_name: str,
                      map: dict[str, Station]) -> float:
    current_station = map[current_station_name]
    end_station = map[end_station_name]
    lat_distance = current_station.position[0] - end_station.position[0]
    lon_distance = current_station.position[1] - end_station.position[1]
    return max(abs(lat_distance), abs(lon_distance))