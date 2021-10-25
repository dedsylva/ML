#! /usr/bin/python3

import sys
from linear import LinearRegression
import numpy as np

if __name__ == '__main__':

	LEARNING_RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
	SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 100 
	ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 10 


	np.random.seed(0)
	X_TRAIN = np.random.rand(SIZE,1)
	X_TRAIN[0] = 0 # that bias
	Y_TRAIN = 2 + 3 * X_TRAIN + np.random.rand(SIZE,1)

	X_TEST = np.random.rand(SIZE,1)
	X_TEST[0] = 0 # that bias
	Y_TEST = 2 + 3 * X_TEST + np.random.rand(SIZE,1)

	lr = LinearRegression(X_TRAIN, Y_TRAIN) 
	lr.regress(ITERATIONS, printable=True, lr=LEARNING_RATE)

	predictions = lr.predict(X_TEST, Y_TEST)

	# Evaluate model
	lr.get_rmse(predictions)
	lr.get_rsquared(predictions)
