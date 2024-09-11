import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
import sys
import evaluation
import os

no_clauses = int(sys.argv[1])
T = int(sys.argv[2])
s = float(sys.argv[3])
no_epochs = int(sys.argv[4])
literal_budget = int(sys.argv[5])
dataset = sys.argv[6]

X_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_test.npy')
no_classes = len(set(Y_train))
number_of_literals = 2*len(X_train[0])
print('# %s dimensions: %d training samples, %d test samples, %d classes, %d literals' %(dataset, len(Y_train), len(Y_test), no_classes, number_of_literals))

tm = MultiClassTsetlinMachine(no_clauses, T, s, weighted_clauses=False)
tm.max_included_literals=literal_budget

all_accuracy = []
all_num_inc = []

# training
print("\nAccuracy over %d epochs:\n" %(no_epochs))
for i in range(no_epochs):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	number_of_includes = evaluation.IncludeCount(tm)

	print("#%d Test accuracy: %.2f%% Training: %.2fs Testing: %.2fs Number of Includes: %d" % (i+1, result, stop_training-start_training, stop_testing-start_testing, number_of_includes))
	all_accuracy.append(result)
	all_num_inc.append(number_of_includes)

# TM log files
if not os.path.exists(r"log"):
	os.makedirs(r"log")
evaluation.ClauseExpr(r"log/clause_expr.txt", tm) 	# clause expressions
evaluation.GetTA_action(r"log/actions.txt", tm) 	# TA actions
evaluation.TA_states(r"log/states.txt", tm) 		# TA states

# accuracy and number of includes during training
result_file = open(r"log/AccuracyVsIncludes.txt", "w")
for each1, each2 in zip(all_accuracy, all_num_inc):
    result_file.write("%.2f %d;\n" %(each1, each2))
result_file.close()