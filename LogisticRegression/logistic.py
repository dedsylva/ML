#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

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


	def predict(self, X_TEST, Y_TEST):
		self.X_TEST = X_TEST
		self.Y_TEST = Y_TEST
		results = self.sigmoid(np.dot(X_TEST, self.weights))
		classes = [1 if r >= 0.5 else 0 for r in results]

		right, wrong = 0., 0.
		ones, zeros = 0., 0.
		for i in range(len(classes)):
			#print('Predicted: %d, Got: %d' %(classes[i], self.Y_TEST[i]))
			if classes[i] == self.Y_TEST[i]:
				right += 1
				if classes[i] == 1:
					ones += 1
				else:
					zeros += 1
			else:
				wrong += 1

		right /= len(classes)
		wrong /= len(classes)

		print('Got %.2f Right, %.2f Wrong' % (right, wrong))
		print('Got %d Ones, %d Zeros' % (ones, zeros))
	
		return results 

	def get_confusion_matrix(self, predictions):
		#  *** CONFUSION MATRIX ***
		#					Predicted
		# Actual		1			0
		#	1      |  TP    FN
		#	0			 |  FP    TN

		confusion = np.zeros((2,2))
		classes = [1 if p >= 0.5 else 0 for p in predictions]

		for i,cl in enumerate(classes):
			if cl == 1:
				if self.Y_TEST[i] == 1:
					confusion[0,0] += 1
				else:
					confusion[1,0] += 1
			else: 
				if self.Y_TEST[i] == 1:
					confusion[0,1] += 1
				else:
					confusion[1,1] += 1

		print('Confusion Matrix')
		print(confusion)
		return confusion


	def get_precision(self, predictions):
		TP, FP = 0., 0.

		classes = [1 if p >= 0.5 else 0 for p in predictions]

		for i,cl in enumerate(classes):
			if cl == 1:
				if self.Y_TEST[i] == 1:
					TP += 1
				else:
					FP += 1
			else:
				continue

		precision = (TP/(TP+FP))*100
		print('Got %.2f of Precision' %(precision))
		return precision


	def get_recall(self, predictions):
		TP, FN = 0., 0.

		classes = [1 if p >= 0.5 else 0 for p in predictions]

		for i,cl in enumerate(classes):
			if cl == 1:
				if self.Y_TEST[i] == 1:
					TP += 1
				else:
					continue
			else:
				if self.Y_TEST[i] == 1:
					FN += 1
				else:
					continue

		recall = (TP/(TP+FN))*100
		print('Got %.2f of Recall' %(recall))
		return recall 

	def get_AUC(self, predictions):
		# thresholds
		thresholds = np.array(range(0,105,5))/100

		TPR = np.zeros((len(thresholds)))
		FPR = np.zeros((len(thresholds)))
		TP = np.zeros((len(thresholds)))
		FP = np.zeros((len(thresholds)))
		FN = np.zeros((len(thresholds)))
		TN = np.zeros((len(thresholds)))

		for i,th in enumerate(thresholds):
			classes = [1 if p >= th else 0 for p in predictions]
			for cl in classes:
				if cl == 1:
					if self.Y_TEST[i] == 1:
						TP[i] += 1	
					else:
						FP[i] += 1	
				else: 
					if self.Y_TEST[i] == 1:
						FN[i] += 1	
					else:
						TN[i] += 1 	

		# True Positive Rate aka Sensitivity
		TPR = TP / (TP + FN)

		# False Positive Rate ( 1 - Sensitivity)
		FPR = FP / (TN + FP)

		plt.scatter(FPR, TPR, label='ROC')
		plt.plot([0,1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend()
		plt.show()

		AUC = round(abs(np.trapz(FPR, TPR)),4)*100
		print('Got %f of AUC' %(AUC))

	def plot_loss(self):

		plt.plot(self.losses, label='Loss')
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.legend()
		plt.show()
