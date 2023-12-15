import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
df = pd.read_csv('findpath_time.csv')

# 提取五个变量的数据
A1 = df['A1']
A2 = df['A2']
A3 = df['A3']
B = df['B']
D = df['D']

# 绘制折线图
plt.plot(A1, label='A1')
plt.plot(A2, label='A2')
plt.plot(A3, label='A3')
plt.plot(B, label='B')
plt.plot(D, label='D')

# 添加标题和标签
plt.title('折线图')
plt.xlabel('次数')
plt.ylabel('变量值')

# 添加图例
plt.legend()

# 显示图形
plt.show()