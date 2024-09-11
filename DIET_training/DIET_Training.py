import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time
import sys
import evaluation
import LiteralExcluding
import os

no_clauses = int(sys.argv[1])
T = int(sys.argv[2])
s = float(sys.argv[3])
exclude_every_no_epochs = int(sys.argv[4])
literal_budget = int(sys.argv[5])
dataset = sys.argv[6]

X_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_test.npy')
no_classes = len(set(Y_train))
number_of_literals = 2*len(X_train[0])

tm = MultiClassTsetlinMachine(no_clauses, T, s, weighted_clauses=False)
tm.max_included_literals=literal_budget

all_accuracy = []
all_num_inc = []

# literal excluding by modifying TA states
InsigLit = LiteralExcluding.InsignificantLiterals(r"log/clause_expr.txt")
new_all_states = LiteralExcluding.Prune(InsigLit, r"log/states.txt", number_of_literals, no_clauses)
tm.fit(X_train, Y_train, epochs=1, incremental=False)
state_list = tm.get_state()
for i in range(tm.number_of_classes):
	for j in range(len(new_all_states[i])):
		state_list[i][1][j] = new_all_states[i][j]
tm.set_state(state_list)
result = 100*(tm.predict(X_test) == Y_test).mean()
all_accuracy.append(result)
number_of_includes = evaluation.IncludeCount(tm)
all_num_inc.append(number_of_includes)
print("After excluding, Accuracy: %.2f%%, Number of Includes: %d" %(result, number_of_includes))

# TM log files after excluding
evaluation.ClauseExpr(r"log/excluded_expr.txt", tm) 	# clause expressions

# re-train
print("\nAccuracy over %d epochs re-training:\n" %(exclude_every_no_epochs))
for i in range(exclude_every_no_epochs):
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
evaluation.ClauseExpr(r"log/clause_expr.txt", tm) 	# clause expressions
evaluation.GetTA_action(r"log/actions.txt", tm) 	# TA actions
evaluation.TA_states(r"log/states.txt", tm) 		# TA states
evaluation.XY4verification(X_test, Y_test, tm, r"log")	# samples for verification

# accuracy and number of includes during training
result_file = open(r"log/AccuracyVsIncludes.txt", "a")
for each1, each2 in zip(all_accuracy, all_num_inc):
    result_file.write("%.2f %d;\n" %(each1, each2))
result_file.close()