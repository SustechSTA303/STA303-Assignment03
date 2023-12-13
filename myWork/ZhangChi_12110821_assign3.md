# Artificial Intelligence-Homework 3

**Name：** 张弛（ZHANG Chi）

**SID：** 12110821

**Link of pull request: **[Zhang Chi 12110821 ](https://github.com/SustechSTA303/STA303-Assignment02/pull/32)



## Introduction
In this assignment, we were asked to find the shortest path from given two stations in London railway by using different algorithms. The algorithms I used consist of `BFS`,`Dijkstra`,`Bellman-Ford`,`Greedy BFS`,`A*` and `Bidirectional A*`. And I used `number of iterations`and  `path length` to test the performance of these algorithms. The result shows that `A*`(with cost defined by Euclidean distance and heuristics defined by Euclidean or Haversine distance) is the best algorithm to find the shortest path in London railway.

-   Algorithm discription

	-   BFS
	-   Dijkstra's Algorithm (cost)
		-   Initialize distances from the start node to all other nodes as infinity.
		-   Set the distance  to the start node as 0.
		-   For each neighbor of the  current node, update their distances if a shorter path (depends on **cost  function**) is found.
		-   Continue until the destination is reached.
	-   Bellman-Ford Algorithm (heuristic)
		-   Initialize distances from the start node to all other nodes as infinity.
		-   Set the distance to the start node as 0.
		-   Repeat the following process for (number of nodes - 1) iterations: For each edge in  the graph, update the distance (calculated in **heuristic function**) to the destination node if a shorter path is found.
		-   Check for negative cycles: For  each edge in the graph, If updating the distance to the destination node results in a shorter path, a negative cycle exists.
		-   The final distances represent the shortest paths from the start node to all other nodes.
	-   Greedy Best First Search (heuristic)
		-   Instead of searching in all direction,  in each step, it will take a step in the direction where **heuristic function** is the smallest.
	-   A* Algorithm (cost, heuristic)
		-   Similar to Dijkstra's  algorithm but uses a **heuristic function** to estimate the cost to reach the destination from the current node.
		-   The A* algorithm combines the  actual cost (from Dijkstra's) and the heuristic cost to make better decisions.
	-   Bidirctional_A* Algorithm (cost, heuristic)
		-   Similarm to A*, but it can search for the shortest path from start_station and from end_station
		-   Searching until  two paths meet

-   For cost and heuristic functions, I have 3 different  measurement method and 6 distinct combinations

	-   Functions
	    
		| Name                | Description                                                  |
		| ------------------- | ------------------------------------------------------------ |
		| Euclidean Distance  | Straight-line  distance between two stations.                |
		| Haversine  Distance | 球面距离或大圆距离：在球面坐标系下，经纬度表示地球表面上的点的位置。要计算两点之间的实际距离，可以使用球面三角法 |
		| 1（only for cost）  | 地铁两站之间本身时间短，都相差不大。但是地铁在两站之间经历了加速-匀速-减速的过程，每一站的走走停停耗费时间很多，所以设置为1，是为了减少站点数量。 |



##  Experiment



## Conclusion



## Appendix

