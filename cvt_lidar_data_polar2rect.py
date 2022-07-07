
import math
import numpy as np
import pandas as pd

from sklearn import linear_model, datasets
from matplotlib import pyplot as plt

data = pd.read_csv('lidar_bag.csv')
data.head()

angle = data['angle']
dist = data['distance']

angle1 = angle.values.reshape(-1,1)
dist1 = dist.values.reshape(-1,1)

# print(angle1, dist1)

x = []
y = []

for i in range(len(angle1)):
    x.append(dist1[i] * math.cos(angle1[i]*math.pi/180))
    y.append(dist1[i] * math.sin(angle1[i]*math.pi/180))

x = np.array(x)
y = np.array(y)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

plt.plot(x,y,'o')
plt.show()
