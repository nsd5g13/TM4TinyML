import numpy as np
import pathlib
import os, random

# -- Encode raw features into one-hot codes -------------------------------------------------
def onehot_encoding(X_raw_features, no_bins):
	thres = []
	for i in range(len(X_raw_features[0])):
		thres.append([])
		for j in range(1, no_bins):
			thres[i].append(np.quantile(X_raw_features[:,i], float(1/no_bins)*j))
	onehot = []

	for i in range(no_bins):
		onehot.append([])
		for j in range(no_bins):
			if i != j:
				onehot[i].append(0)
			else:
				onehot[i].append(1)
	onehot=onehot[::-1]

	X_bool=[]
	for i in range(len(X_raw_features)):
		X_bool.append([])
		for j, each in enumerate(X_raw_features[i]):
			tmp = thres[j]
			tmp = tmp + [each]
			tmp.sort()
			X_index = tmp.index(each)
			if no_bins == 2:
				if X_index == 1:
					X_bool[i].append(1)
				else:
					X_bool[i].append(0)
			else:
				X_bool[i].extend(onehot[X_index])
	
	return X_bool, thres

# -- Encode raw features into thermometer codes -------------------------------------------------
def thermo_encoding(X_raw_features, no_bins):
	thres = []
	for i in range(len(X_raw_features[0])):
		thres.append([])
		for j in range(1, no_bins):
			thres[i].append(np.quantile(X_raw_features[:,i], float(1/no_bins)*j))
	thermo = []

	for i in range(no_bins):
        	thermo.append([])
        	for j in range(no_bins):
                	if i > j:
                        	thermo[i].append(0)
                	else:
                        	thermo[i].append(1)
	thermo=thermo[::-1]

	X_bool=[]
	for i in range(len(X_raw_features)):
		X_bool.append([])
		for j, each in enumerate(X_raw_features[i]):
			tmp = thres[j]
			tmp = tmp + [each]
			tmp.sort()
			X_index = tmp.index(each)
			if no_bins == 2:
				if X_index == 1:
					X_bool[i].append(1)
				else:
					X_bool[i].append(0)
			else:
				X_bool[i].extend(thermo[X_index])
	
	return X_bool

# -- load EMG dataset ---------------------------------------------------------
def EMG_load(dataset_file_path):
	all_subjects = os.walk(dataset_file_path)
	all_dirs = [x[0] for x in all_subjects]
	all_subjects = os.walk(dataset_file_path)
	all_txt = [x[2] for x in all_subjects]
	all_features = []
	all_labels = []
	cnt = 0
	for dir, txt in zip(all_dirs[1:], all_txt[1:]):
		for each in txt:
			file_name = dir + '/' + each
			f = open(file_name, 'r')
			lines = f.readlines()
			f.close()

			print('Read text file: ' + file_name)
			all_features.append([])
			all_labels.append([])
			prev_timepoint = 0
			for each_line in lines[1:]:
				features_label = each_line.split()
				features = [float(x) for x in features_label[:-1]]
				label = features_label[-1]
				curr_timepoint = features[0]
				if curr_timepoint == prev_timepoint + 1:
					all_labels[cnt].append(int(label))
					all_features[cnt].append(features[1:])
				else:	# interpolation
					no_interpolation = int(curr_timepoint - prev_timepoint)
					for i in range(no_interpolation):
						all_labels[cnt].append(int(label))
						all_features[cnt].append(features[1:])
				prev_timepoint = curr_timepoint
			cnt = cnt + 1	

	return all_features, all_labels

# -- relabel unmarked data, if a sample is found in the marked dataset, for EMG dataset
def EMG_relabel(X_Class0, X_notClass0, Y_notClass0):
	Y_relabelled = []
	for each_unmarked in X_Class0:
		FoundOrNot = np.all(X_notClass0==each_unmarked, axis=1)
		matched_indices = np.where(FoundOrNot)
		if np.size(matched_indices, axis=None) != 0:
			new_labels = Y_notClass0[matched_indices]
			new_labels = new_labels.tolist()
			new_label = max(set(new_labels), key=new_labels.count)
			Y_relabelled.append(new_label)
		else:
			Y_relabelled.append(0)
	Y_relabelled = np.array(Y_relabelled)
	return Y_relabelled

# -- load Sports dataset ---------------------------------------------------------
def Sports_load(dataset_file_path):
	all_subjects = os.walk(dataset_file_path)
	all_dirs = [x[0] for x in all_subjects]
	all_dirs.sort(key=len)
	all_dirs = all_dirs[20:]
	all_subjects = os.walk(dataset_file_path)
	all_txt = [x[2] for x in all_subjects]
	all_txt = all_txt[2]
	all_features = []
	all_labels = []
	cnt = 0
	for dir in all_dirs:
		idx1 = dir.index('Sports/a')
		idx2 = dir.index('/p')
		label = dir[idx1 + 8 : idx2]
		label = int(label) - 1
		for txt in all_txt:
			file_name = dir + '/' + txt
			f = open(file_name, 'r')
			lines = f.readlines()
			f.close()

			print('Read text file: ' + file_name)
			all_features.append([])
			all_labels.append([])			
			for each_line in lines:
				features = each_line.replace('\n', '').split(',')
				features = [float(x) for x in features]
				all_features[cnt].append(features)
				all_labels[cnt].append(label)
			cnt = cnt + 1	

	return all_features, all_labels

# -- load Human Activity dataset ---------------------------------------------------------
def HAR_load(dataset_file_path):
	TrainOrTest = dataset_file_path.split('/')[2]
	file_name = dataset_file_path + '/X_' + TrainOrTest + '.txt'
	f = open(file_name ,'r')
	X_lines = f.readlines()
	f.close()

	file_name = dataset_file_path + '/y_' + TrainOrTest + '.txt'
	f = open(file_name ,'r')
	y_lines = f.readlines()
	f.close()		

	utilized_feature_indices = [0, 1, 2, 9, 10, 11, 40, 41, 42]

	all_features = []
	all_labels = []
	for X_line, y_line in zip(X_lines, y_lines):
		features = X_line.replace('\n', '').split()
		features = [features[i] for i in utilized_feature_indices]
		features = [float(x) for x in features]
		all_features.append(features)
		label = y_line.replace('\n', '')
		label = int(label) - 1
		all_labels.append(label)	

	return all_features, all_labels

# -- load Gesture Phase Segmentation dataset ---------------------------------------------------------
def Gesture_load(dataset_file_path):
	labels = ['Rest', 'Preparation', 'Stroke', 'Hold', 'Retraction']
	all_subjects = os.walk(dataset_file_path)
	all_csv = [x[2] for x in all_subjects][0]
	all_raw_csv = []
	all_features = []
	all_labels = []
	for each in all_csv:
		if 'raw' in each:
			all_raw_csv.append(each)
	for each in all_raw_csv:
		file_name = dataset_file_path + '/' + each
		f = open(file_name, 'r')
		lines = f.readlines()
		f.close()

		for each_line in lines[1:]:
			features_label = each_line.replace('\n', '').split(',')
			features = [float(x) for x in features_label[:-2]]
			label = features_label[-1]
			label = labels.index(label)
			all_features.append(features)
			all_labels.append(label)

	return all_features, all_labels

# -- load Gas Sensor Array Drift dataset ---------------------------------------------------------
def Gas_load(dataset_file_path):
	all_subjects = os.walk(dataset_file_path)
	all_bat = [x[2] for x in all_subjects][0]
	all_features = []
	all_labels = []
	for each in all_bat:
		file_name = dataset_file_path + '/' + each
		f = open(file_name, 'r')
		lines = f.readlines()
		f.close()

		for each_line in lines[1:]:
			features_label = each_line.replace('\n', '').split()
			label = int(features_label[0])-1
			all_labels.append(label)
			features = [float(x.split(":")[1]) for x in features_label[1:]]
			all_features.append(features)

	return all_features, all_labels

# -- load Statlog Vehicle Silhouettes dataset ---------------------------------------------------------
def Statlog_load(dataset_file_path):
	labels = ['van', 'saab', 'bus', 'opel']
	all_subjects = os.walk(dataset_file_path)
	all_bat = [x[2] for x in all_subjects][0]
	all_features = []
	all_labels = []
	for each in all_bat[2:]:
		file_name = dataset_file_path + '/' + each
		f = open(file_name, 'r')
		lines = f.readlines()
		f.close()

		for each_line in lines:
			features_label = each_line.replace('\n', '').split()
			label = labels.index(features_label[-1])
			all_labels.append(label)
			features = [int(x) for x in features_label[:-1]]
			all_features.append(features)

	return all_features, all_labels

# -- load Mammographic Mass dataset ---------------------------------------------------------
def Mammography_load(dataset_file_path):
	f = open(dataset_file_path, 'r')
	lines = f.readlines()
	f.close()

	all_features = []
	all_labels = []
	for each_line in lines:
		features_label = each_line.replace('\n', '').split(',')
		label = int(features_label[-1])
		all_labels.append(label)
		all_features.append(features_label[:-1])
	return all_features, all_labels

# -- onehot/thermometer encoding for Mammographic Mass dataset -----------------------------------------------
def Mammography_encoding(all_features, no_bins, coding):
	no_features = len(all_features[0])
	unmissing_features = [[] for i in range(no_features)]
	for features in all_features:
		for i in range(no_features):
			if features[i] != '?':
				unmissing_features[i].append(int(features[i]))

	thres = [[] for i in range(no_features)]
	for i in range(no_features):
		for j in range(1, no_bins):
			thres[i].append(np.quantile(unmissing_features[i], float(1/no_bins)*j))

	encoded = []
	for i in range(no_bins):
		encoded.append([])
		for j in range(no_bins):
			if coding == 'onehot':
				if i != j:
					encoded[i].append(0)
				else:
					encoded[i].append(1)
			elif coding == 'thermometer':
				if i > j:
                        		encoded[i].append(0)
				else:
                        		encoded[i].append(1)
			else:
				print('The given coding method %s is not recognized' %coding)
	encoded=encoded[::-1]

	X_bool = []
	for features in all_features:
		bool_features = []
		for i, feature in enumerate(features):
			if feature != '?':
				feature = int(feature)
				tmp = thres[i]
				tmp = tmp + [feature]
				tmp.sort()
				X_index = tmp.index(feature)
			else:
				random.seed(1)
				X_index = random.randint(0, no_bins)
			if no_bins == 2:
				if X_index == 1:
					bool_features.append(1)
				else:
					bool_features.append(0)
			else:
				bool_features.extend(encoded[X_index])
		X_bool.append(bool_features)
	return X_bool

# -- load Sensorless Drive Diagnosis dataset ---------------------------------------------------------
def Sensorless_load(dataset_file_path):
	f = open(dataset_file_path, 'r')
	lines = f.readlines()
	f.close()

	all_features = []
	all_labels = []
	for each_line in lines:
		features_label = each_line.replace('\n', '').split()
		label = int(features_label[-1]) - 1
		all_labels.append(label)
		features = [float(x) for x in features_label[:-1]]
		all_features.append(features)
	return all_features, all_labels				
	
					
# -- Perform certain pooling method for each given number of timesteps --------------------------
def pooling(all_features, all_labels, winLength, poolMethod):
	X = []
	Y = []
	for subject_features, subject_labels in zip(all_features, all_labels):
		subject_X = []
		subject_Y = []

		no_windows = int(np.floor((len(subject_features)-winLength)/winLength))
		start_time = 0
		end_time = start_time + winLength
		for window in range(no_windows):
			subject_window = subject_features[start_time:end_time]
			subject_window = np.array(subject_window)
			if poolMethod == 'max':
				pool_values = subject_window.max(axis=0)
			elif poolMethod == 'avg':
				pool_values = np.mean(subject_window, axis=0)
			elif poolMethod == 'rms':		# root mean square
				tmp = np.square(subject_window)
				tmp = np.sum(tmp, axis=0)
				pool_values = np.sqrt(tmp)
			elif poolMethod == 'mav':		# mean absolute value
				tmp = np.abs(subject_window)
				pool_values = np.mean(tmp, axis=0)				
			else:
				print('The specified pooling method is not recognized!')

			window_labels = subject_labels[start_time:end_time]
			window_label = max(set(window_labels), key=window_labels.count)
			
			subject_Y.append(window_label)
			subject_X.append(pool_values.tolist())			

			start_time = start_time + winLength
			end_time = end_time + winLength	

		X.append(subject_X)
		Y.append(subject_Y)
	return X, Y

# -- Generate features based on given number of timesteps (window length) and stride --------------------------
def SlidingWindow(all_features, all_labels, window_length, stride):
	X = []
	Y = []
	cnt = 0
	for subject_features, subject_labels in zip(all_features, all_labels):
		no_windows = int(np.floor((len(subject_labels)-window_length)/stride) + 1)
		start_time = 0
		end_time = start_time + window_length
		for window in range(no_windows):
			X.append([])
			for i in range(window_length):
				X[cnt].extend(subject_features[start_time+i])

			window_labels = subject_labels[start_time:end_time]
			window_label = max(set(window_labels), key=window_labels.count)
			Y.append(window_label)
			
			start_time = start_time + stride
			end_time = start_time + window_length
			cnt = cnt + 1
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

# -- Split dataset into training/test set
def DatasetSplit(X, Y, TrainingSetRatio):
	random.seed(1)
	n = random.sample(range(len(Y)), len(Y))
			
	X_train = []
	Y_train = []
	for each in n[0:int(len(Y)*TrainingSetRatio)]:
		X_train.append(X[each])
		Y_train.append(Y[each])

	X_test = []
	Y_test = []
	for each in n[int(len(Y)*TrainingSetRatio):]:
		X_test.append(X[each])
		Y_test.append(Y[each])

	X_train=np.array(X_train)
	Y_train=np.array(Y_train)
	X_test=np.array(X_test)
	Y_test=np.array(Y_test)

	return X_train, Y_train, X_test, Y_test