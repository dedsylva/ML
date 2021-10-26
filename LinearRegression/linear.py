#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class LinearRegression:
	def __init__(self, X, Y, lr=0.001, optimizer='GD'):
		self.X = X
		self.Y = Y
		self.lr = lr
		self.optimizer = optimizer

	def mse(self):
		# Tomar cuidado com o np.sum para dados de dimensoes maiores
		cost = np.sum((self.prediction - self.Y)**2) 
		return cost/(2*self.X.shape[0])

	def gradient_descent(self):
		residuals = (self.prediction - self.Y)
		self.weights -= np.dot(self.X.T, residuals)*(self.lr/self.X.shape[0])
		return 

	def Adam(self, b1=0.9, b2=0.999, eps=1e-8):
		residuals = (self.prediction - self.Y)
		gradients = np.dot(self.X.T, residuals)*(self.lr/self.X.shape[0])

		t = 0
		m_w = np.zeros_like(self.weights)
		v_w = np.zeros_like(self.weights)

		t += 1
		m_w = b1 * m_w + (1 - b1) * gradients # estimates mean of gradient
		v_w = b2 * v_w + (1 - b2) * gradients**2 # estimates variance of gradient

		mhat_w = m_w / (1. - b1**t) # corrects bias towards zero
		vhat_w = v_w / (1. - b2**t)

		self.weights -= self.lr * mhat_w / (np.sqrt(vhat_w) + eps) # updating weights
		return


	def regress(self, iterations, printable=True):
		# Random weights at the beginnin
		self.weights = np.zeros((self.X.shape[1], self.Y.shape[1]))

		# Gathering data for plot
		self.losses = []
		for it in range(iterations):
			# Predict
			self.prediction = np.dot(self.X, self.weights)

			# Calculates loss (aka mean squared error)
			loss = self.mse()
			self.losses.append(loss)

			# Optimizer
			if self.optimizer == 'Adam':
				self.Adam()

			elif self.optimizer == 'GD':
				self.gradient_descent()
			else:
				raise ValueError('Invalid Optimizer. Optimions are Adam or GD, but got %s instead' %(self.optimizer))

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

		print('RMSE score of model:     %.4f' %(rmse))
		return	


	def get_rsquared(self, pred):
		# Sum of errors if taking the mean as if model is true
		sst = np.sum((self.Y_TEST - np.mean(self.Y_TEST))**2)

		# Sum of the square of residuals
		ssr = np.sum((pred- self.Y_TEST)**2)

		# r2 score
		r2 = 1 - (ssr/sst)

		print('R Square score of model: %.4f' %(r2))
		return


	def plot(self):
		plt.plot(self.losses, label='Loss')
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()
