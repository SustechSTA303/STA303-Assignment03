import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open('output.txt') as file:
    lines = file.readlines()

data_list1 = []
data_list2 = []
data_list3 = []

for i in range(len(lines)):
    if lines[i].startswith('Algorithm: astar'):
        if lines[i+1].startswith('Distance function: haversine'):
            data_list1.append(float(lines[i+2].split(':')[1].strip()))
    elif lines[i].startswith('Algorithm: ucs'):
        if lines[i+1].startswith('Distance function: haversine'):
            data_list2.append(float(lines[i+2].split(':')[1].strip()))
    elif lines[i].startswith('Algorithm: bfs'):
        if lines[i+1].startswith('Distance function: haversine'):
            data_list3.append(float(lines[i+2].split(':')[1].strip()))

df = pd.DataFrame({'astar': data_list1,'ucs': data_list2,'bfs': data_list3})

print(data_list1)
print(data_list2)
print(data_list3)
print(len(data_list2))

plt.figure(figsize=(8,5))
sns.kdeplot(data=df,fill = True)
plt.title('Distance Function: Haversine')
plt.xlabel('Total Cost')
plt.ylabel('Probability Density')
plt.show()