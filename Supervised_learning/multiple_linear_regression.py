import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Regressor import My_multiple_regression


X, y = load_diabetes(return_X_y=True)

# print(X.shape)


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(r2_score(y_test, y_pred))

print(reg.coef_)
print(reg.intercept_)


reg1 = My_multiple_regression()
reg1.fit(X_train, y_train)
print(reg1.coef_[1:])
print(reg1.intercept_)
print(r2_score(y_test,reg1.predict(X_test)))

