#! /usr/bin/python3

import sys
from logistic import LogisticRegression
import numpy as np
import sklearn.datasets

if __name__ == '__main__':

	LEARNING_RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
	ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 100 
	MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 20 

	# Loading Dataset ready
	iris = sklearn.datasets.load_iris()

	X_TRAIN = iris.data[MAX:,:2]
	Y_TRAIN = ((iris.target != 0 ) * 1).reshape(-1,1)[MAX:]

	X_TEST = iris.data[:MAX,:2]
	Y_TEST = ((iris.target != 0 ) * 1).reshape(-1,1)[:MAX]

	# Logistic Prediction
	lr = LogisticRegression(X_TRAIN, Y_TRAIN, LEARNING_RATE) 
	weights = lr.regress(ITERATIONS, printable=False)
	predictions = lr.predict(X_TEST, Y_TEST, printable=True)
	
	lr.plot()
