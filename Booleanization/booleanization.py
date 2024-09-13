import os, sys, random
import numpy as np
import preprocessing

from sklearn import datasets

# dataset name as the argument
if sys.argv[1].lower() == 'all':
	all_datasets = ['emg', 'iris', 'sports', 'har', 'digits', 'gesture', 'gas', 'statlog', 'mammography', 'sensorless']
else:
	all_datasets = [sys.argv[1].lower()]

# make directory to store the booleanized dataset(s)
if not os.path.exists(r"bool_datasets"):
	os.makedirs(r"bool_datasets")

for dataset in all_datasets:
	match dataset:

		# ------------- IRIS ----------------------------------------------------
		case "iris":
			iris = datasets.load_iris()
			X, _ = preprocessing.onehot_encoding(iris.data, 3)
			Y = iris.target
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# ------------- Digits ----------------------------------------------------
		case "digits":
			digits = datasets.load_digits()
			X = np.where(digits.data > 7.5, 1, 0)
			Y = digits.target
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)		

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
			
			# exclude samples of class 0 from quantile binning process, as class 0 indicates unmarked data
			X_notClass0 = X[Y != 0]
			Y_notClass0 = Y[Y != 0]
			X_Class0 = X[Y == 0]
								
			X_notClass0, thresholds = preprocessing.onehot_encoding(X_notClass0, 2)
			
			# relabel unmarked data
			print('Relabelling unmarked data ... ')
			thresholds = sum(thresholds, [])
			X_Class0 = (X_Class0 > thresholds).astype(int)
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

			X, _ = preprocessing.onehot_encoding(X, 2)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)
	
		# --------------- Human acitivity ------------------------------------------------
		case "har":
			X_train, Y_train = preprocessing.HAR_load(r'TinyML_raw_dataset/HAR/train')
			X_test, Y_test = preprocessing.HAR_load(r'TinyML_raw_dataset/HAR/test')

			X_train = np.array(X_train)
			X_test = np.array(X_test)
			X = np.concatenate((X_train, X_test), axis=0)
			
			X, _ = preprocessing.onehot_encoding(X, 2)
			X_train = X[:len(Y_train)]
			X_test = X[len(Y_train):]

		# --------------- Gesture phase segmentation --------------------------------
		case "gesture":
			X, Y = preprocessing.Gesture_load(r'TinyML_raw_dataset/Gesture')
			X = np.array(X)
			Y = np.array(Y)
			X, _ = preprocessing.onehot_encoding(X, 8)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Gas Sensor Array Drift ----------------------------------------
		case "gas":
			X, Y = preprocessing.Gas_load(r'TinyML_raw_dataset/Gas')
			X = np.array(X)
			Y = np.array(Y)
			X, _ = preprocessing.onehot_encoding(X, 2)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Statlog Vehicle Silhouettes ----------------------------------------
		case "statlog":
			X, Y = preprocessing.Statlog_load(r'TinyML_raw_dataset/Statlog')
			X = np.array(X)
			Y = np.array(Y)
			X = preprocessing.thermo_encoding(X, 20)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Mammographic Mass ------------------------------------------------
		case "mammography":
			X, Y = preprocessing.Mammography_load(r'TinyML_raw_dataset/Mammography/mammographic_masses.data')
			X = preprocessing.Mammography_encoding(X, 3, 'onehot')
			X = np.array(X)
			Y = np.array(Y)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)

		# --------------- Sensorless Drive Diagnosis ------------------------------------------------
		case "sensorless":
			X, Y = preprocessing.Sensorless_load(r'TinyML_raw_dataset/Sensorless/Sensorless_drive_diagnosis.txt')
			X = np.array(X)
			Y = np.array(Y)
			X, _ = preprocessing.onehot_encoding(X, 3)
			X_train, Y_train, X_test, Y_test = preprocessing.DatasetSplit(X, Y, 0.8)						
														
		case _:
			print("The given dataset %s is not recognized." %sys.argv[1])

	# store datasets in given directory
	if not os.path.exists(r"bool_datasets/"+dataset):
		os.makedirs(r"bool_datasets/"+dataset)
	np.save(r"bool_datasets/"+dataset+'/X_train.npy', X_train)
	#np.savetxt(r"bool_datasets/"+dataset+'/X_train.txt', X_train, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/Y_train.npy', Y_train)
	#np.savetxt(r"bool_datasets/"+dataset+'/Y_train.txt', Y_train, fmt='%d')
	np.save(r"bool_datasets/"+dataset+'/X_test.npy', X_test)
	np.save(r"bool_datasets/"+dataset+'/Y_test.npy', Y_test)