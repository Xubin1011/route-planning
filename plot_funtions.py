import numpy as np
import matplotlib.pyplot as plt
import math


# x = np.linspace(0, 8, 1000)
# y = np.where(x <= 4.5, np.exp(x) - np.exp(4.5), -100)
# plt.plot(x, y)
# plt.xlabel('Driving time(in h)')
# plt.ylabel('Reward')
# plt.title('Reward for driving time')
# plt.grid(True)
# plt.show()



# x = np.linspace(-1, 5, 1000)
# # y = np.where(x<0, 0 ,np.where(x <= 0.75, np.exp(5 * x) - np.exp(3.75), - 10 * (np.exp(1.5 * x) - np.exp(1.125))))
# y = np.where(x<0, 0 ,np.where(x <= 0.75, np.exp(5 * x) - np.exp(3.75), -32* x + 24))
# plt.plot(x, y)
# plt.xlabel('Charging time(in h)')
# plt.ylabel('Reward')
# plt.title('Reward for charging time')
# plt.grid(True)
# plt.show()


# x = np.linspace(-1, 0.75, 1000)
# y = np.where(x > 0, -2 * (np.exp(5 * x) - 1), -100)
# plt.plot(x, y)
# plt.xlabel('Rest action in parking lots(%) * Remaining rest time (in h)')
# plt.ylabel('Reward')
# plt.title('Reward for rest time in parking lots')
# plt.grid(True)
# plt.show()

# x = np.linspace(-0.2, 0.2, 1000)
# y = np.where(x < 0, -6, np.where(x <= 0.1, np.log(0.1 * x) + 5, 0.4))
# plt.plot(x, y)
# plt.xlabel('Remaining SoC')
# plt.ylabel('Reward')
# plt.title('Reward for energy ')
# plt.grid(True)
# plt.show()

# x = np.linspace(-25, 25, 1000)
# # y = np.where(x < 0, (-1 * np.exp(-0.15 * x  ) + 1)/1000, (np.exp(0.1 * x ) - 1) / 1000)
# # y = np.where(x < 0, (-1 * np.exp(-1 * x) + 1)/1000, (np.exp(1 * x ) - 1) / 1000)
# y = x
# plt.plot(x , y)
# plt.xlabel('Distance traveled forward(in km)')
# plt.ylabel('Reward')
# plt.title('Reward for distance ')
# plt.grid(True)
# plt.show()

##episolon greedy
x = np.linspace(0, 40000, 5000)
# y = np.where(x<0, 0 ,np.where(x <= 0.75, np.exp(5 * x) - np.exp(3.75), - 10 * (np.exp(1.5 * x) - np.exp(1.125))))
y = 0.05 + (0.9-0.05) * np.exp(-1 * x / 10000)
plt.plot(x, y)
plt.xlabel('step')
plt.ylabel('eps_threshold')
plt.grid(True)
plt.show()

# # y = 0.1 + (0.9-0.1) * np.exp(-1 * x / 3530)
# y = -20000 / np.log(1/8)
# print(y)

