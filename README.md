## Assignment3: Find the shortest path in underground lines 
### File Description
- [london](london) 
  - [underground_lines.csv](london%2Funderground_lines.csv)(London Underground lines Data)
  - [underground_routes.csv](london%2Funderground_routes.csv)(Detailed data of London Underground lines)
  - [underground_stations.csv](london%2Funderground_stations.csv)(London Underground stations Data)
- [visualization_underground](visualization_underground)
  - [London_railway.html](visualization_underground%2FLondon_railway.html)(London Underground Route Map)
  - [my_path_in_London_railway.html](visualization_underground%2Fmy_path_in_London_railway.html)(Visualize a certain path on the London Underground route map)
  - [London_Underground_Overground_DLR_Crossrail_map.svg](visualization_underground%2FLondon_Underground_Overground_DLR_Crossrail_map.svg)(London Underground Route Map)
- [build_data.py](build_data.py) (Reading London Underground Line Data)
- [find_shortest_path.py](find_shortest_path.py) (Find the shortest path between two stations)
- [plot_underground_lines.py](plot_underground_lines.py) (Draw a map of the London Underground route)
- [plot_underground_path.py](plot_underground_path.py) (Draw a path on the London Underground route map)
- [README.md](README.md)
### How to visualize the map and path
- To draw a map of the London Underground route:
```
python plot_underground_lines.py
```
and open [London_railway.html](visualization_underground%2FLondon_railway.html) to view it.
- To draw path on the map, such as ['Acton Town', 'Chiswick Park', 'Turnham Green', 'Stamford Brook']:
```
python plot_underground_path.py
```
Then, you can open [my_path_in_London_railway.html](visualization_underground%2Fmy_path_in_London_railway.html) to view it.

### Assignment3 Requirements
- Implement the function get_path() in [find_shortest_path.py](find_shortest_path.py).
- After implementation, you can visualize the shortest path at a given starting and ending station, by running the following command: 
```
python find_shortest_path.py  "Acton Town"  "Turnham Green"
```
The above example specifies the path from Acton Town to Turnham Green. 

***(Note: If there are blank space in the station name, station name need to be wrapped in double quotation marks("") in the command line.)***

Then, open `visualization_underground/my_shortest_path_in_London_railway.html` to view the path.