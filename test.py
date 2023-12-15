import timeit
import argparse
from build_data import build_data
from plot_underground_path import plot_path
from find_shortest_path import (
    dijkstra,
    bellman_ford,
    spfa,
    floyd_warshall,
    get_path,
)

def calculate_path_length(path, stations):
    length = 0
    for i in range(1, len(path)):
        station1 = stations[path[i - 1]]
        station2 = stations[path[i]]
        # Assuming equal cost for all edges, you may adjust this based on your actual distance calculation
        length += 1
    return length

def run_algorithm(algorithm_func, start_station, end_station, stations):
    start_time = timeit.default_timer()
    path = algorithm_func(start_station, end_station, stations)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    path_length = calculate_path_length(path, stations)
    return path, execution_time, path_length

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("start_station_name", type=str, help="start_station_name")
    parser.add_argument("end_station_name", type=str, help="end_station_name")
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    # 构建站点数据
    stations, underground_lines = build_data()

    # 运行并测量每个算法所需的执行时间
    algorithms = [
        ("Dijkstra 算法", dijkstra),
        ("Bellman-Ford 算法", bellman_ford),
        ("SPFA 算法", spfa),
        ("Floyd-Warshall 算法", floyd_warshall),
        ("A* 算法", get_path),
    ]

    for algorithm_name, algorithm_func in algorithms:
        print(f"\n运行 {algorithm_name}...")
        path, execution_time, path_length = run_algorithm(algorithm_func, start_station_name, end_station_name, stations)
        print(f"最短路径: {path}")
        print(f"路径长度: {path_length}")
        print(f"执行时间: {execution_time:.6f} 秒")
        plot_path(path, f"visualization_underground/{start_station_name}_to_{end_station_name}_{algorithm_name}_path.html", stations, underground_lines)
