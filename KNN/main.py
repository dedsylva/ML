import sys
import numpy as np
from knn import KNN

if __name__ == '__main__':
	if len(sys.argv) > 1:
		k = int(sys.argv[1])
	else:
		k = 3

	print('%d Nearest Neighbors' %(k))
	SIZE = 10
	data = np.random.randint(0, 100, size=SIZE)
	classes = np.random.randint(0, 3, size=SIZE)
	#classes = np.zeros((SIZE,))
	print(data)
	print(classes)

	knn = KNN(data, classes, k)
	x = 3
	cl = knn.get_class(x)	
	print('Class of input %d: %d' %(x, cl))
