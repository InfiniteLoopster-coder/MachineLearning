import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 


def prepare_dataset():
    X,y=make_regression(n_samples=100,n_features=1,random_state=42,noise=20)
    return X,y

class Error:
    def __init__(self, y_true, y_pred):
        self.y_train = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        assert self.y_train.shape[0]==self.y_pred.shape[0],"Dimension of the array must be same "
        
        
    def mae(self):
        """
        Mean Squared Error (MSE)
        MSE = (1/n) * Σ (y_true_i - y_pred_i)^2
        """
        result = 0
        for i in range(self.y_train.shape[0]):
            result =  result + abs(self.y_train[i] - self.y_pred[i])
        return result
    
    def mse(self):
        return np.mean((self.y_train-self.y_pred)**2)
    

    def rmse(self):
        return (np.mean((self.y_train-self.y_pred)**2))**0.5


    def r_2_score(self):
        y_mean = np.mean(self.y_train)
        num = (self.y_train - self.y_pred)**2
        den = (self.y_train - y_mean)**2
        r2 = 1 - np.sum(num)/np.sum(den)
        return r2
    
    def adjusted_r_2(self, n, k):
        y_mean = np.mean(self.y_train)
        num = (self.y_train - self.y_pred)**2
        den = (self.y_train - y_mean)**2
        r2 = 1 - np.sum(num)/np.sum(den)
        num1 = (n - 1)
        den1 = (n - k - 1)
        adj_r2 = 1-(1 - r2) * np.sum(num1)/np.sum(den1)
        return adj_r2



class My_multiple_regression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        self.coef_ = np.linalg.inv(np.dot(X_train.T, X_train))@X_train.T@y_train
        self.intercept_=self.coef_[0]
        return self

    def predict(self, X_test):
        X_test = np.insert(X_test, 0,1, axis=1)
        return np.dot(X_test,self.coef_)















class MyLinearRegression():
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X, y):
        X=X.flatten()

        X_mean = np.mean(X)
        y_mean = np.mean(y)
        print(X[:5])
        print(X_mean)
        result=X-X_mean
        print(result[:5])
        self.m=np.sum((X-X_mean)*(y-y_mean))/np.sum((X-X_mean)**2)
        self.b = y_mean - self.m*X_mean

    def predict(self,X):
        X=X.flatten()
        return self.m*X+self.b



if __name__=="__main__":
    X,y=prepare_dataset()
    lr=MyLinearRegression()
    lr.fit(X,y)
    y_pred=lr.predict(X)
    print(f"Value of m:{lr.m},b={lr.b}")
    print(f"R2 Score of Model is : {r2_score(y,y_pred)}")
    lr1=LinearRegression()
    lr1.fit(X,y)
    y_pred1=lr1.predict(X)
    print(f"Value of m:{lr1.coef_},b={lr1.intercept_}")
    print(f"R2 Score of Model is : {r2_score(y,y_pred1)}")
    assert r2_score(y,y_pred) == r2_score(y,y_pred1),"Koi to jhol hai re baba 🤣"
    # plt.scatter(X,y)
    # plt.plot(X,y_pred)
    # plt.show()
    


    
    