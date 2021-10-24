import sys
import numpy as np
from knn import KNN

if __name__ == '__main__':

	k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
	DIMENSION = int(sys.argv[2]) if len(sys.argv) > 2 else 2
	N_CLASSES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
	SIZE = 10

	print('%d Nearest Neighbors with %d dimensions' %(k, DIMENSION))
	data = np.random.randint(0, 20, size = (SIZE,DIMENSION))
	classes = np.random.randint(0, N_CLASSES, size =SIZE)

	knn = KNN(data, classes, k)
	x = np.random.randint(0,20, size=DIMENSION) 
	cl = knn.get_class(x)	
	print('Class of input: %d' %(cl))
