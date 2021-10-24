import numpy as np

class KNN:
	def __init__(self, data, classes, k=3):
		self.data = data
		self.classes = classes
		self.k = k

	def distance(self,x1,x2):
		dist = 0
		for i in range(len(x1)):
			dist += (x1[i] - x2[i])**2
		return np.sqrt(dist)

	def get_class(self, x):
		dis = np.zeros((self.data.shape[0],)).astype('float')

		for i in range(self.data.shape[0]):
			dis[i] = self.distance(self.data[i], x)

		return self.classes[np.argmin(dis)] 
