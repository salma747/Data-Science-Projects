import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]
plt.hist(X.iloc[:, 1])
imputer = Imputer(missing_values=np.NAN, strategy='mean', axis=0)
imputer.fit(X.iloc[:, [1, 2]])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])


labelEncoderY=LabelEncoder()
Y= labelEncoderY.fit_transform(Y)

labelEncoderX=LabelEncoder()
X.iloc[:,0]=labelEncoderX.fit_transform(X.iloc[:,0])

oneHotEncoder=OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

