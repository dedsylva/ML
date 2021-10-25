import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class KNN:
  def __init__(self, k=3):
    self.k = k
    self.MAX_DISTANCE = 1e7

  def distance(self,x1,x2):
    dist = 0
    for i in range(len(x1)):
      dist += (x1[i] - x2[i])**2
    return np.sqrt(dist)

  def get_class(self, data, classes, x):
    self.data = data
    self.classes = classes

    dis = np.zeros((self.data.shape[0],)).astype('float')

    for i in range(self.data.shape[0]):
      dis[i] = self.distance(self.data[i], x)

    cl = []
    indexes = []
    for i in range(self.k):
      index = np.argmin(dis)
      indexes.append(index)
      cl.append(self.classes[index])
      dis[index] = self.MAX_DISTANCE

    if self.data.shape[0] > self.k:
      # saving the k nearest neighbors for special plotting
      self.k_data = np.zeros((len(indexes), self.data.shape[1]))
      for i in range(len(indexes)):
        self.k_data[i] = self.data[indexes[i]]

      self.k_classes = [self.classes[i] for i in indexes]

      self.data = np.delete(self.data, indexes, axis=0)
      self.classes = np.delete(self.classes, indexes, axis=0)   


    # decide which class input x is
    return int(stats.mode(cl)[0][0])


  def plot_classes(self, x, cl):
    colors = ['blue' if c == 1 else 'red' for c in self.classes]
    color_x = 'blue' if cl == 1 else 'red'
    k_colors = ['blue' if c == 1 else 'red' for c in self.k_classes]
    
    d1 = plt.scatter(self.data[:,0], self.data[:,1], c=colors, marker="o", s=50)
    d2 = plt.scatter(x[0], x[1], c=color_x, marker="*", s=100)
    d3 = plt.scatter(self.k_data[:,0], self.k_data[:,1], c=k_colors, marker="d", s=50)

    plt.legend((d1, d2, d3), ('Dataset', 'Input', 'K-Nearest Neighbors'),
          loc = 'lower left',
          fontsize = 8)
