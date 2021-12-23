from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

x = iris.data
y = iris.target

k_range = range(1,31)
k_skor = []

for i in k_range:
knn = KNeighborsClassifier(n_neighbors = i)
skor = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
k_skor.append(skor.mean())

plt.plot(k_range, k_skor)
plt.xlabel('K argumen')
plt.ylabel('K Skor')

plt.show()