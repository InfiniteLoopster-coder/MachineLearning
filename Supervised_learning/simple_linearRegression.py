from Regressor import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('placement.csv')
X = df.iloc[: ,0].values
y = df.iloc[: ,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)





