import sys
import numpy as np
from knn import KNN
import matplotlib.pyplot as plt

if __name__ == '__main__':

	k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
	SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 10
	N_CLASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
	DIMENSION = 2

	print('%d Nearest Neighbors with size %d and %d classes' 
		%(k, SIZE, N_CLASSES))
	
	knn = KNN(k)
	data = np.zeros((1, DIMENSION))
	classes = np.zeros((1,))

	for i in range(SIZE):
		d = np.random.randint(0, 20, size = (1,DIMENSION))
		c = np.random.randint(0, N_CLASSES, size=1)
		data = np.append(data, d.reshape(-1,1).T, axis=0)
		classes = np.append(classes, np.array(c, ndmin=1))

		if i == 0:
			data = np.delete(data, 0, axis=0)
			classes = np.delete(classes, 0, axis=0)


		x = np.random.randint(0,20, size=DIMENSION) 
		cl = knn.get_class(data, classes, x)
		color = 'blue' if cl == 1 else 'red'

		print('Class of input: (%d, %s)' %(cl, color))	
		
		if i > k:
			knn.plot_classes(x, cl)

		# appending input
		data = np.append(data, x.reshape(-1,1).T, axis=0)
		classes = np.append(classes, np.array(cl, ndmin=1))

		plt.show()
