# python==3.12.6

import os, sys, random
import numpy as np
import preprocessing

from sklearn import datasets
from keras.datasets import mnist

# dataset name as the argument
if sys.argv[1].lower() == 'all':
	all_datasets = ['emg', 'iris', 'sports', 'har', 'digits', 'gesture', 'gas', 'statlog', 'mammography', 'sensorless', 'mnist']
else:
	all_datasets = [sys.argv[1].lower()]

# make directory to store the booleanized dataset(s)
if not os.path.exists(r"datasets"):
	os.makedirs(r"datasets")

for dataset in all_datasets:
	match dataset:
		# ------------- MNIST --------------------------------------------------
		case "mnist":
			(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
			X_train = X_train.reshape(X_train.shape[0], 28*28) 
			X_test = X_test.reshape(X_test.shape[0], 28*28)
		
		# ------------- IRIS ----------------------------------------------------
		case "iris":
			iris = datasets.load_iris()
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(iris.data, iris.target, 0.8)

		# ------------- Digits ----------------------------------------------------
		case "digits":
			digits = datasets.load_digits()
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(digits.data, digits.target, 0.8)		

		# --------------- EMG ------------------------------------------------
		case "emg":
			PoolingWindow = 100
			PoolingMethod = 'rms'
			SlidingWinLen = 20
			Stride = 3

			print('Loading EMG dataset from given path:')
			X, Y = preprocessing.EMG_load(r'TinyML_raw_dataset/EMG')
			X, Y = preprocessing.pooling(X, Y, PoolingWindow, PoolingMethod)
			X, Y = preprocessing.SlidingWindow(X, Y, SlidingWinLen, Stride)
			
			X_notClass0 = X[Y != 0]
			Y_notClass0 = Y[Y != 0]
			X_Class0 = X[Y == 0]
								
			Y_relabelled = preprocessing.EMG_relabel(X_Class0, X_notClass0, Y_notClass0)
			X = np.concatenate((X_Class0, X_notClass0), axis=0)
			Y = np.concatenate((Y_relabelled, Y_notClass0), axis=0)
			
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Sports ------------------------------------------------
		case "sports":
			PoolingWindow = 10
			PoolingMethod = 'avg'
			SlidingWinLen = 1
			Stride = 1

			print('Loading Sports dataset from given path:')
			X, Y = preprocessing.Sports_load(r'TinyML_raw_dataset/Sports')
			X, Y = preprocessing.pooling(X, Y, PoolingWindow, PoolingMethod)
			X, Y = preprocessing.SlidingWindow(X, Y, SlidingWinLen, Stride)

			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)
	
		# --------------- Human acitivity ------------------------------------------------
		case "har":
			X_train, Y_train = preprocessing.HAR_load(r'TinyML_raw_dataset/HAR/train')
			X_test, Y_test = preprocessing.HAR_load(r'TinyML_raw_dataset/HAR/test')

			X_train = np.array(X_train)
			X_test = np.array(X_test)
			X = np.concatenate((X_train, X_test), axis=0)
			
			X_train = X[:len(Y_train)]
			X_test = X[len(Y_train):]

		# --------------- Gesture phase segmentation --------------------------------
		case "gesture":
			X, Y = preprocessing.Gesture_load(r'TinyML_raw_dataset/Gesture')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Gas Sensor Array Drift ----------------------------------------
		case "gas":
			X, Y = preprocessing.Gas_load(r'TinyML_raw_dataset/Gas')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Statlog Vehicle Silhouettes ----------------------------------------
		case "statlog":
			X, Y = preprocessing.Statlog_load(r'TinyML_raw_dataset/Statlog')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Mammographic Mass ------------------------------------------------
		case "mammography":
			X, Y = preprocessing.Mammography_load(r'TinyML_raw_dataset/Mammography/mammographic_masses.data')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Sensorless Drive Diagnosis ------------------------------------------------
		case "sensorless":
			X, Y = preprocessing.Sensorless_load(r'TinyML_raw_dataset/Sensorless/Sensorless_drive_diagnosis.txt')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)						
														
		case _:
			print("The given dataset %s is not recognized." %sys.argv[1])

	# store datasets in given directory
	if not os.path.exists(r"datasets/"+dataset):
		os.makedirs(r"datasets/"+dataset)
	np.save(r"datasets/"+dataset+'/X_train.npy', X_train)
	#np.savetxt(r"datasets/"+dataset+'/X_train.txt', X_train, fmt='%d')
	np.save(r"datasets/"+dataset+'/Y_train.npy', Y_train)
	#np.savetxt(r"datasets/"+dataset+'/Y_train.txt', Y_train, fmt='%d')
	np.save(r"datasets/"+dataset+'/X_test.npy', X_test)
	np.save(r"datasets/"+dataset+'/Y_test.npy', Y_test)