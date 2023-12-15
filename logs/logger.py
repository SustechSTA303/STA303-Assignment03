import os
import csv


class logger:
    def __init__(self, algorithm, heuristic):
        self.algorithm = algorithm
        self.heuristic = heuristic
        self.path = "logs/results.csv"

        self.distance = []
        self.time = []

        print(f"Test for algorithm:{algorithm}, heuristic:{heuristic}")

    def update(self, distance, time):
        self.distance.append(distance)
        self.time.append(time)

    def log(self):
        path = self.path
        for i in range(10):
            if os.path.exists(path):
                with open(path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.algorithm, self.heuristic, i, self.distance[i], self.time[i]])
            else:
                columns = ["Algorithm", "Heuristic", "test_id", "distance", "time"]
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(columns)
                    writer.writerow([self.algorithm, self.heuristic, i, self.distance[i], self.time[i]])
