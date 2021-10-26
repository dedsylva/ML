#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class LinearRegression:
	def __init__(self, X, Y, lr=0.001):
		self.X = X
		self.Y = Y
		self.lr = lr

	def mse(self):
		# Tomar cuidado com o np.sum para dados de dimensoes maiores
		cost = np.sum((self.prediction - self.Y)**2) 
		return cost/(2*self.X.shape[0])

	def gradient_descent(self):
		residuals = (self.prediction - self.Y)

		self.weights -= np.dot(self.X.T, residuals)*(self.lr/self.X.shape[0])

		return 

	def regress(self, iterations, printable=True):
		# Random weights at the beginnin
		self.weights = np.zeros((self.X.shape[1], self.Y.shape[1]))

		for it in range(iterations):
			# Predict
			self.prediction = np.dot(self.X, self.weights)

			# Calculates loss (aka mean squared error)
			loss = self.mse()

			# Optimizer
			self.gradient_descent()

			if (printable):
				print('Iteration %d, loss: %.2f' %(it, loss))
	
			if loss == 1e-02:
				break

		return self.weights


	def predict(self, X_TEST, Y_TEST):
		self.X_TEST = X_TEST
		self.Y_TEST = Y_TEST
		return np.dot(X_TEST, self.weights)


	def get_rmse(self, pred):
		rmse = np.sum((pred- self.Y_TEST)**2) 
		rmse = np.sqrt(rmse/self.X.shape[0])

		print('RMSE score of model:     %.2f' %(rmse))
		return	


	def get_rsquared(self, pred):
		# Sum of errors if taking the mean as if model is true
		sst = np.sum((self.Y_TEST - np.mean(self.Y_TEST))**2)

		# Sum of the square of residuals
		ssr = np.sum((pred- self.Y_TEST)**2)

		# r2 score
		r2 = 1 - (ssr/sst)

		print('R Square score of model: %.2f' %(r2))
		return
