import csv
import os


class Station:
    """
    The Station class contains four attributes: id, name, position, and links.
    Position is a binary combination of longitude and latitude
    and links are a list of stations adjacent to the Station object
    """
    def __init__(self, id, name, position):
        self.id = id
        self.name = name
        self.position = position
        self.links = set()


def build_data():
    """
    builds the 'map' by reading the data files
    Returns:
        station(dict[str, Station]): A mapping between station names and station objects of the name
        underground_lines(dict[str, dict]): A mapping between underground lines name and a dictionary containing relevant
                                            information about underground lines
    """
    stations = {}
    underground_lines = {}
    rootdir = os.path.dirname(__file__)
    r = csv.reader(open(os.path.join(rootdir, 'london/underground_stations.csv')))
    next(r)  # jump the first line
    for record in r:
        id = int(record[0])
        lat = float(record[1])
        lon = float(record[2])
        name = record[3]
        stations[id] = Station(id, name, (lat, lon))

    r = csv.reader(open(os.path.join(rootdir, 'london/underground_routes.csv')))
    next(r)  # jump the first line
    for id1, id2, lineNumber in r:
        id1 = int(id1)
        id2 = int(id2)
        stations[id1].links.add(stations[id2])
        stations[id2].links.add(stations[id1])
        lineNumber = int(lineNumber)
        if lineNumber not in underground_lines:
            underground_lines[lineNumber] = {'lat': [stations[id1].position[0], stations[id2].position[0], None],
                                             'lon': [stations[id1].position[1], stations[id2].position[1], None],
                                             'stations': {stations[id1].name, stations[id2].name}}
        else:
            underground_lines[lineNumber]['lat'].extend([stations[id1].position[0], stations[id2].position[0], None])
            underground_lines[lineNumber]['lon'].extend([stations[id1].position[1], stations[id2].position[1], None])
            underground_lines[lineNumber]['stations'].add(stations[id1].name)
            underground_lines[lineNumber]['stations'].add(stations[id2].name)
    r = csv.reader(open(os.path.join(rootdir, 'london/underground_lines.csv')))
    next(r) # jump the first line
    for lineNumber, name, colour, stripe in r:
        lineNumber = int(lineNumber)
        underground_lines[lineNumber]['name'] = name
        underground_lines[lineNumber]['colour'] = colour.upper()
        underground_lines[lineNumber]['stripe'] = stripe
    stations = {v.name: v for k, v in stations.items()}
    underground_lines = {v['name']: v for k, v in underground_lines.items()}
    return stations, underground_lines

