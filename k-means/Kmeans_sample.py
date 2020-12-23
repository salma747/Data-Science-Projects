import numpy as np
import pandas as ps
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

data=ps.read_csv('Social_Network_Ads.csv')
X=data.iloc[:,[2,3]].values
Y=data.iloc[:,4].values
X_train ,X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

kmeans=KMeans(n_clusters=2)
kmeans.fit(X_train)
cluster_centers=kmeans.cluster_centers_
plt.scatter(X_train[:,0],X_train[:,1],s=50,color='orange')
plt.scatter( cluster_centers[0,0], cluster_centers[0,1],s=150,color='green',marker='s')
plt.scatter(cluster_centers[1,0], cluster_centers[1,1],s=150,color='red',marker='s')
plt.show()

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
               np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, kmeans.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                       alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
      plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                          c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kmeans')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()