
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
dataset = pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,-1].values
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)
Z_pred=regressor.predict(X)
Y_pred=regressor.predict([[6.5]])
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth of bluff(SVR)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()
sc_X=StandardScaler()
sc_y=StandardScaler()
