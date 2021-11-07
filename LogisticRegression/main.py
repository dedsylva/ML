#! /usr/bin/python3

import sys
from logistic import LogisticRegression
import sklearn.datasets
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

	LEARNING_RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
	ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 5000 
	TEST_SIZE = float(sys.argv[3]) if len(sys.argv) > 3 else .2 

	# Loading Dataset ready
	iris = sklearn.datasets.load_iris()

	X = iris.data[:,:2]
	Y = ((iris.target != 0 ) * 1).reshape(-1,1)

	X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=TEST_SIZE, random_state=0)

	'''
	MAX = 20
	X_TRAIN = iris.data[MAX:,:2]
	Y_TRAIN = ((iris.target != 0 ) * 1).reshape(-1,1)[MAX:]

	X_TEST = iris.data[:MAX,:2]
	Y_TEST = ((iris.target != 0 ) * 1).reshape(-1,1)[:MAX]
	'''


	# Logistic Prediction
	lr = LogisticRegression(X_TRAIN, Y_TRAIN, LEARNING_RATE) 
	weights = lr.regress(ITERATIONS, printable=False)
	predictions = lr.predict(X_TEST, Y_TEST)

	confusion = lr.get_confusion_matrix(predictions)
	precision = lr.get_precision(predictions)
	recall = lr.get_recall(predictions)
	lr.plot_AUC(predictions)
