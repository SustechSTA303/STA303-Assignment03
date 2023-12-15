from algorithms.build_algorithms import build_algorithms
from build_data import build_data
from evaluation.utils import distance

dijkstra = build_algorithms("dijkstra")
bellman_ford = build_algorithms("bellman ford")
dfs = build_algorithms("deep first search")
bfs = build_algorithms("best first search")
astar = build_algorithms("A star")

stations, underground_lines = build_data()

path = dijkstra("Acton Town", "Beckton", stations)
print(f"Dijkstra: {path}")
print(f"Distance: {distance(path, stations)}")

path = dfs("Acton Town", "Beckton", stations)
print(f"Deep First Search: {path}")
print(f"Distance: {distance(path, stations)}")

path = bellman_ford("Acton Town", "Beckton", stations)
print(f"Bellman Ford: {path}")
print(f"Distance: {distance(path, stations)}")

path = bfs("Acton Town", "Beckton", stations, 'euclidean')
print(f"Best First Search with euclidean: {path}")
print(f"Distance: {distance(path, stations)}")

path = bfs("Acton Town", "Beckton", stations, 'manhattan')
print(f"Best First Search with manhattan: {path}")
print(f"Distance: {distance(path, stations)}")

path = bfs("Acton Town", "Beckton", stations, 'diagonal')
print(f"Best First Search with diagonal: {path}")
print(f"Distance: {distance(path, stations)}")

path = astar("Acton Town", "Beckton", stations, 'euclidean')
print(f"Astar Search with euclidean: {path}")
print(f"Distance: {distance(path, stations)}")

path = astar("Acton Town", "Beckton", stations, 'manhattan')
print(f"Astar Search with manhattan: {path}")
print(f"Distance: {distance(path, stations)}")

path = astar("Acton Town", "Beckton", stations, 'diagonal')
print(f"Astar Search with diagonal: {path}")
print(f"Distance: {distance(path, stations)}")
