#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class LogisticRegression:
	def __init__(self, X, Y, lr=0.001):
		self.X = X
		self.Y = Y
		self.lr = lr

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def cross_entropy(self):
		return (-self.Y * np.log(self.prediction) - (1-self.Y) * np.log(1-self.prediction)).mean()

	def gradient_descent(self):
		residuals = (self.prediction - self.Y)
		self.weights -= np.dot(self.X.T, residuals)*(self.lr/self.X.shape[0])
		return 

	def regress(self, iterations, printable=True):
		# Zero Weights at the beginning 
		self.weights = np.zeros((self.X.shape[1], self.Y.shape[1]))

		# Gathering data for plot
		self.losses = []
		for it in range(iterations):
			# Predict
			self.prediction = self.sigmoid(np.dot(self.X, self.weights))

			# Calculates loss (aka mean squared error)
			loss = self.cross_entropy()
			self.losses.append(loss)

			self.gradient_descent()

			if (printable):
				print('Iteration %d, loss: %.2f' %(it, loss))
	
			if loss == 1e-02:
				break

		return self.weights


	def predict(self, X_TEST, Y_TEST, printable=True):
		self.X_TEST = X_TEST
		self.Y_TEST = Y_TEST

		predicted = self.sigmoid(np.dot(X_TEST, self.weights))
		classes = [1 if p >= 0.5 else 0 for p in predicted]
		
		right = 0.
		wrong = 0.
	
		for i in range(len(classes)):
			print('Predicted: %d | True: %d | Probability: %.1f %%' %(classes[i], self.Y_TEST[i], predicted[i]*100))
			if classes[i] == self.Y_TEST[i]:
				right += 1
		
		right /= len(classes)
		wrong = 1 - right

		# TODO: Create AUC, Confusion Matrix, Precision, Recall Metrics
		print('Got %.2f Right, %.2f Wrong' %(right, wrong))

	def plot(self):
		# TODO: Plot the distribution of the classes, similar to what we did in KNN

		plt.plot(self.losses, label='Loss')
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()
