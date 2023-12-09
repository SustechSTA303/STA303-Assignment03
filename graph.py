import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Creating a DataFrame with the results
data = {
    "Algorithm": ["A* (Euclidean)", "A* (Manhattan)", "BFS", "DFS", "Dijkstra"],
    "Total Duration (s)": [14.488, 12.958, 2.598, 3.359, 8.944],
    "Total Path Length (km)": [21683.588, 21704.616, 22535.308, 46495.224, 25320.684]
}

df = pd.DataFrame(data)

# Define a Morandi color palette
morandi_palette = ["#C0AEBD", "#A5A8B1", "#B4ABAC", "#B1B5A8", "#A8B0B1"]  # Morandi colors

# Plotting Total Duration with Morandi colors
plt.figure(figsize=(10, 6))
sns.barplot(x="Algorithm", y="Total Duration (s)", data=df, palette=sns.color_palette("PuBu"))
plt.title("Total Duration of Algorithms")
plt.ylabel("Total Duration (seconds)")
plt.xlabel("Algorithm")
#plt.xticks(rotation=45)
plt.show()

# Plotting Total Path Length with Morandi colors
plt.figure(figsize=(10, 6))
sns.barplot(x="Algorithm", y="Total Path Length (km)", data=df, palette=sns.color_palette("PuBu"))
plt.title("Total Path Length of Algorithms")
plt.ylabel("Total Path Length (kilometers)")
plt.xlabel("Algorithm")
#plt.xticks(rotation=45)
plt.show()
