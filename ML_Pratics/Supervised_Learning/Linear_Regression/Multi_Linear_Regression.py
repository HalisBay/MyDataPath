import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,2)
coef = np.array([3,7])
y = 10 + np.dot(x,coef)

# fig = plt.figure()
# ax = fig.add_subplot(111,projection = "3d")
# ax.scatter(x[:,0],x[:,1],y)
# plt.show()

lr = LinearRegression()
lr.fit(x,y)

fig = plt.figure()
ax = fig.add_subplot(111,projection = "3d")
ax.scatter(x[:,0],x[:,1],y)

x1, x2 = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))
y_pred = lr.predict(np.column_stack([x1.flatten(), x2.flatten()]))
ax.plot_surface(x1,x2,y_pred.reshape(x1.shape),alpha = 0.2)
plt.show()
