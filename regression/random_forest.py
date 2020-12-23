import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
Y=dataset.iloc[:, 2].values
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
regressor.fit(X ,Y)
Z_pred=regressor.predict(X)
Y_pred=regressor.predict([[6.5]])                     
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,Color= 'magenta')
plt.plot(X_grid,regressor.predict(X_grid),color='green')
plt.title('Truth or bluff (Random Forest Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,Color= 'magenta')
plt.plot(X_grid,regressor.predict(X_grid),color='black')
plt.title('Truth or bluff (Random Forest Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()
