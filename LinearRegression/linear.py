#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class LinearRegression:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def mse(self):
		# Tomar cuidado com o np.sum para dados de dimensoes maiores
		cost = np.sum((np.dot(self.weights.T, self.X) - self.Y)**2) 
		return cost/(2*self.X.shape[0])

	def gradient_descent(self, lr):
		dc_dweights = (np.dot(self.weights.T, self.X) - self.Y)*lr

		self.weights -= dc_dweights*self.X/self.X.shape[0]

		return self.weights 

	def regress(self, iterations, printable=True, lr=0.001):
		# Random weights at the beginning
		self.weights = np.random.rand(self.X.shape[0], self.Y.shape[0])

		for it in range(iterations):
			# Calculates loss (aka mean squared error)
			loss = self.mse()

			# Optimizer
			self.weights = self.gradient_descent(lr)

			if (printable):
				print('Iteration %d, loss: %.2f' %(it, loss))
	
			if loss == 1e-02:
				break

	def predict(self, X_TEST, Y_TEST):
		self.X_TEST = X_TEST
		self.Y_TEST = Y_TEST
		return (np.dot(self.weights.T, X_TEST))


	def get_rmse(self, pred):
		rmse = np.sum((pred- self.Y_TEST)**2) 
		rmse = np.sqrt(rmse/self.X.shape[0])

		print('RMSE score of model: %.2f' %(rmse))
		return	


	def get_rsquared(self, pred):
		# Sum of errors if taking the mean as if model is true
		sst = np.sum((self.Y_TEST - np.mean(self.Y_TEST))**2)

		# Sum of the square of residuals
		ssr = np.sum((pred- self.Y_TEST)**2)

		# r2 score
		r2 = 1 - (ssr/sst)

		print('sst', sst)
		print('ssr', ssr)
		print('R Square of model: %.2f' %(r2))
		return
