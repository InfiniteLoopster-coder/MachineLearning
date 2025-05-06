class LinearRegression():
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):

        num = 0
        den = 0

        for i in range(X_train.shape[0]):
            # sum up to 1 to n (x-x^)(y-y^)
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean()))
            # sum up to 1 to n (x - x^)^2
            den = den + ((X_train[i] - X_train.mean())*(X_train - X_train.mean()))

        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, X_test):
        return self.m * X_test + self.b
    
    