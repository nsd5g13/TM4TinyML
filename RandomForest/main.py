import numpy as np
import sys, random
from Forest import Forest

from sklearn import datasets

dataset = sys.argv[1]
no_trees = int(sys.argv[2])
max_depth = int(sys.argv[3])

X_train = np.load(r'datasets/datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'datasets/datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'datasets/datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'datasets/datasets/'+dataset+'/Y_test.npy')

forest = Forest(max_depth=max_depth, no_trees=no_trees,
                min_samples_split=2, min_samples_leaf=1,
                feature_search=5, bootstrap=True)

X_train_new = []
Y_train_new = []
for X, Y in zip(X_train, Y_train):
	if '?' not in X:
		X_train_new.append(X.tolist())
		Y_train_new.append(Y.tolist())
X_train = np.array(X_train_new)
Y_train = np.array(Y_train_new)

X_test_new = []
Y_test_new = []
for X, Y in zip(X_test, Y_test):
	if '?' not in X:
		X_test_new.append(X.tolist())
		Y_test_new.append(Y.tolist())
X_test = np.array(X_test_new)
Y_test = np.array(Y_test_new)

print('Number of features: %d' %(len(X_train[0])))
models = forest.train(X_train[0:500], Y_train[0:500])

train_acc = forest.eval(X_train, Y_train)  # Retrieve train accuracy
test_acc = forest.eval(X_test, Y_test)  # Retrieve test accuracy

print('Train accuracy: %.2f%%' %(train_acc*100))
print('Test accuracy: %.2f%%' %(test_acc*100))

# save model
f = open(r'micropython_input/RF.txt', 'w')
for i in range(no_trees):
	model = models[i]
	info = str(model).split()

	node_idx = []
	leaf = []
	for j, each in enumerate(info):
		if 'True' in each or 'False' in each:
			node_idx.append(j)
			if 'True' in each:
				leaf.append(True)
			else:
				leaf.append(False)

	tree = [[] for i in range(max_depth+1)]
	sprout_idx = [0 for i in range(max_depth+1)]
	for idx, is_leaf in zip(node_idx, leaf):
		if is_leaf == True:
			depth = int(info[idx+1])
			tree[depth].append(int(info[idx+2]))
			tree[depth].append('label')
			tree[depth].append('nan')
			tree[depth].append('nan')
		else:
			depth = int(info[idx+1])
			tree[depth].append(int(info[idx+3]))
			tree[depth].append(float(info[idx+4]))
			tree[depth].append(sprout_idx[depth])
			tree[depth].append(sprout_idx[depth]+4)
			sprout_idx[depth] = sprout_idx[depth] + 8

	tree = [x for x in tree if x != []]

	for each_depth in tree:
		for each in each_depth:
			f.write('%s ' %each)
		f.write('\n')

	f.write('\n')		

f.close()

# test samples
np.savetxt(r'micropython_input/X.txt', X_test, fmt='%f', delimiter=' ')
predict_Y = forest.predict(X_test)
f = open(r"micropython_input/Y.txt", 'w')
f.write('predict\tactual\n')
for predict, actual in zip(predict_Y, Y_test):
	f.write('%d\t%d\n' %(predict, actual))
f.close()

# save result
f = open(r'log/results.txt', 'a')
f.write("%d\t%d\t%.2f\n" %(no_trees, max_depth, test_acc*100))
f.close()	
	