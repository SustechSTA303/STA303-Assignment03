from typing import List
import random

from build_data import Station, build_data


def distance(path: List[str],
             map: dict[str, Station]) -> float:
    dist = 0
    i = 0
    while i < len(path) - 1:
        station1 = map[path[i]]
        station2 = map[path[i + 1]]
        dist += station1.edges[station2.name]
        i = i + 1

    return dist


def generate_test(map: dict[str, Station],
                  file_name: str,
                  test_size: int):
    path = "tests"
    stations = list(map.keys())
    test = []
    for i in range(test_size):
        test_sample = tuple(random.sample(stations, 2))
        if test_sample not in test:
            test.append(test_sample)

    dest = path + "/" + file_name
    with open(dest, 'w') as file:
        for test_sample in test:
            file.write(f"{test_sample[0]}; {test_sample[1]}\n")


def read_test(file_name: str):
    test = []
    with open(file_name, 'r') as file:
        for line in file:
            test_sample = line.strip().split('; ')
            test.append((test_sample[0], test_sample[1]))
    return test


if __name__ == '__main__':
    stations, _ = build_data()

    for i in range(10):
        generate_test(stations, f"test_{i}.txt", 10000)