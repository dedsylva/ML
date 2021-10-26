#! /usr/bin/python3

import sys
from linear import LinearRegression
import numpy as np

if __name__ == '__main__':

	LEARNING_RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
	SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 100 
	FEATURES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
	TARGETS = int(sys.argv[4]) if len(sys.argv) > 4 else 1 
	ITERATIONS = int(sys.argv[5]) if len(sys.argv) > 5 else 10 
	OPTIMIZER = sys.argv[6] if len(sys.argv) > 6 else 'GD'

	print('Linear Regression using %s Optimizer' %(OPTIMIZER))

	np.random.seed(0)
	X_TRAIN = np.random.rand(SIZE,FEATURES)
	X_TRAIN[:,0] = 1 # that bias
	Y_TRAIN = 2 + 3 * X_TRAIN[:,1].reshape(-1,1) + np.random.rand(SIZE,TARGETS)

	X_TEST = np.random.rand(SIZE,FEATURES)
	X_TEST[:,0] = 1 # that bias
	Y_TEST = 2 + 3 * X_TEST[:,1].reshape(-1,1) + np.random.rand(SIZE,TARGETS)

	lr = LinearRegression(X_TRAIN, Y_TRAIN, LEARNING_RATE, OPTIMIZER) 
	weights = lr.regress(ITERATIONS, printable=False)

	predictions = lr.predict(X_TEST, Y_TEST)

	# Evaluate model
	lr.get_rmse(predictions)
	lr.get_rsquared(predictions)

	# Plotting
	lr.plot()
