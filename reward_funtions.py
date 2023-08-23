import numpy as np
import matplotlib.pyplot as plt
import math

# 生成横坐标上的点
x = np.linspace(0, 9, 1000)
w2, w3, w3_1, w4 = 5, 0.1, 5, 0.4

# 计算函数值
y = np.zeros_like(x)
for i in range(len(x)):
    if x[i] >= 4.5:
        y[i] = -10 * ( x[i]-4.5 )
    # elif x[i] < 0.1:
    #     y[i] = np.exp(50 * x[i])/ 100 -1
    #     print(y[i])
    else:
        y[i] = -1 * (4.5-x[i])

# 绘制图像
plt.plot(x, y, linewidth=2)
plt.xlabel('driving_time (in h)')
plt.ylabel('reward')
# plt.title('Function ')
plt.grid(True)
plt.text(0.9, 0.1, f'w7 = 10\nw8 = 1',
         bbox=dict(facecolor='white', edgecolor='black'),
         verticalalignment='bottom', horizontalalignment='right', transform=plt.gca().transAxes)
plt.show()
