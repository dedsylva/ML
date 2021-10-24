import numpy as np

class KNN:
	def __init__(self, data, classes, k=3):
		self.data = data
		self.classes = classes
		self.k = k

	def distance(self,x1,x2):
		return np.sqrt(x1**2 + x2**2)

	def get_class(self, x):
		dis = np.zeros_like(self.data).astype('float')

		for i in range(len(self.data)):
			dis[i] = self.distance(self.data[i], x)

		print(dis, np.argmin(dis))

		return self.classes[np.argmin(dis)]
