from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X,y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20)

# plt.scatter(X,y)
# plt.show()

lr = LinearRegression()

lr.fit(X,y)
# print(lr.coef_)
# print(lr.intercept_)


m = 19.4

class GBRegressor:
    def __init__(self, learning_rate,epochs):
        self.m = m
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_m = -2 * np.sum(y - self.m*X.ravel() - self.b*X.ravel())
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
            #print(loss_slope_b,self.b)
        print(self.m ,self.b)


    def predict(self, X):
        return self.m * X + self.b



gd = GBRegressor(0.001,50)
gd.fit(X,y)
gd.predict(X)


