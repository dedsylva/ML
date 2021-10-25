#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class KNN:
  def __init__(self, N_CLASSES, k=3):
    self.N_CLASSES = N_CLASSES
    self.k = k
    self.MAX_DISTANCE = 1e7

  def distance(self,x1,x2):
    dist = 0
    for i in range(len(x1)):
      dist += (x1[i] - x2[i])**2
    return np.sqrt(dist)

  def get_class(self, data, classes, x, centroid=True):
    self.data = data
    self.classes = classes
    self.CENTROID= centroid

    assert self.classes.shape[0] == self.data.shape[0], 'Number of classes %d diferent than number of samples %d,' %(self.classes.shape[0], self.data.shape[0])
 
    if (self.CENTROID): 
      # Calculating the Centroid of each class
      # For that, creates matrix cl, i.e., for each vector cl[i] (for each class)
      # It has (sum_x, sum_y, number of elements of this class)
      # And then for calculating the centroid, we simply do cl[i] /= cl[i,2]

      self.centr = np.zeros((self.N_CLASSES, 3)) 

      for i in range(self.classes.shape[0]):
        j = self.classes[i]

        self.centr[j,0] += self.data[i,0]
        self.centr[j,1] += self.data[i,1]
        self.centr[j,2] += 1


      # Calculating the Centroid and the distance of input to each one
      dis = np.zeros((self.N_CLASSES,)).astype('float')
      for i in range(self.N_CLASSES):
        self.centr[i] = self.centr[i] / self.centr[i,2] if self.centr[i,2] != 0 else 0
        dis[i] = self.distance(x, self.centr[i])

      return self.classes[np.argmin(dis)]

    else:
      # If no centroid is required, we simply calculate the distance of k-Nearest Neighbors
      # and decide the class of the input

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
    
    
    d1 = plt.scatter(self.data[:,0], self.data[:,1], c=colors, marker="o", s=50)
    d2 = plt.scatter(x[0], x[1], c=color_x, marker="x", s=100)    

    if (self.CENTROID):
      k_centr = ['blue' if c == 1 else 'red' for c in range(self.centr.shape[0])]
      d3 = plt.scatter(self.centr[:,0], self.centr[:,1], c=k_centr, marker="*", s=150)

      plt.legend((d1, d2, d3), ('Dataset', 'Input', 'Centroids'),
            loc = 'lower left',
            fontsize = 8)

    else:
      k_colors = ['blue' if c == 1 else 'red' for c in self.k_classes]
      d3 = plt.scatter(self.k_data[:,0], self.k_data[:,1], c=k_colors, marker="d", s=50)

      plt.legend((d1, d2, d3), ('Dataset', 'Input', 'K-Nearest Neighbors'),
            loc = 'lower left',
            fontsize = 8)
