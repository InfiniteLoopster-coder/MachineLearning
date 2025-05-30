from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


X,y = make_regression(n_samples=4, n_features=1, n_informative=1, n_targets=1, noise=80, random_state=13)

# plt.scatter(X, y)
# plt.show()

reg = LinearRegression()
reg.fit(X,y)
print(reg.coef_)
print(reg.intercept_)

# plt.scatter(X,y)
# plt.plot(X, reg.predict(X), color ='red')
# plt.show()

# lets apply Gradient descent assuming slope is constant m = 78.35
# and let's assume the starting value for intercept b = 0

y_pred = ((78.35 * X) + 0).reshape(4)

# plt.scatter(X,y)
# plt.plot(X, reg.predict(X), color = 'red', label ='OLS')
# plt.plot(X,y_pred, color = '#00a65a', label = 'b = 0')
# plt.legend()
# plt.show()


# m = 78.35
# b = 0

# loss_slope = -2 * np.sum(y - m*X.ravel() -b)

# # print(loss_slope)

# lr = 0.1
# step_size = loss_slope*lr

# print(step_size)

# b = b - step_size

# print(b)

# y_pred1 = ((78.35 * X) + b).reshape(4)

# # plt.scatter(X,y)
# # plt.plot(X, reg.predict(X), color = 'red', label = 'OLS')
# # plt.plot(X,y_pred1, color = '#00a65a', label = 'b = {}'.format(b))
# # plt.plot(X,y_pred, color = '#A3E4D7', label = 'b = 0')
# # plt.legend()
# # plt.show()

# # Iteration 2

# loss_slope2 = -2 * np.sum(y - m*X.ravel() - b)
# print(loss_slope2)

# step_size = loss_slope2*lr


# b2 = b - step_size

# print(b2)

# y_pred2 = ((78.35 * X) + b2).reshape(4)
# # plt.plot(X, reg.predict(X), color = 'red', label = 'OLS')
# # plt.plot(X,y_pred2, color = '#00a65a', label = 'b = {}'.format(b))
# # plt.plot(X,y_pred2, color = '#A3E4D7', label = 'b = 0')
# # plt.legend()
# # plt.show()

# # Iteration 3

# loss_slope3 = -2 * np.sum(y - m*X.ravel() - b)
# print("loss_slope3", loss_slope3)
# step_size = loss_slope3*lr
# print("step_size", step_size)

# b3 = b - step_size
# y_pred3 = ((78.35 * X) + b3).reshape(4)
# plt.plot(X, reg.predict(X), color = 'red', label = 'OLS')
# plt.plot(X,y_pred3, color = '#00a65a', label = 'b = {}'.format(b))
# plt.plot(X,y_pred3, color = '#A3E4D7', label = 'b = 0')
# plt.legend()
# plt.show()






b = -100
m = 78.35
lr = 0.01

epochs = 100

for i in range(epochs):
    loss_slope = -2 * np.sum(y - m*X.ravel() - b)
    b = b - (lr * loss_slope)
    y_pred = m*X + b
    plt.plot(X,y_pred)

plt.scatter(X,y)
plt.show()