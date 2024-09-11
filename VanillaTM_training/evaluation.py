from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np

def IncludeCount(tm):
	number_of_classes = tm.number_of_classes
	number_of_clauses = tm.number_of_clauses
	number_of_features = tm.number_of_features
	number_of_includes = 0

	for i in range(number_of_classes):

		for j in range(0, number_of_clauses, 2):
			for k in range(number_of_features):
				if tm.ta_action(i,j,k) == 1:
					number_of_includes = number_of_includes + 1

		for j in range(1, number_of_clauses, 2):
			for k in range(number_of_features):
				if tm.ta_action(i,j,k) == 1:
					number_of_includes = number_of_includes + 1

	return number_of_includes

def GetTA_action(action_file_name, tm):
	no_classes = tm.number_of_classes
	no_clauses = tm.number_of_clauses
	no_features = tm.number_of_features
	action_file = open(action_file_name, "w")
	for i in range(no_classes):
		for j in range(0, no_clauses, 2):
			for k in range(tm.number_of_features):
				action_file.write("%d" %tm.ta_action(i,j,k))
			action_file.write("\n")

		for j in range(1, no_clauses, 2):
			for k in range(tm.number_of_features):
				action_file.write("%d" %tm.ta_action(i,j,k))
			action_file.write("\n")
	action_file.close()
	
def ClauseExpr(expr_file_name, tm):
	no_classes = tm.number_of_classes
	no_clauses = tm.number_of_clauses
	no_features = tm.number_of_features
	expr_file = open(expr_file_name, "w")
	for i in range(no_classes):
		expr_file.write("\nClass %d Positive Clauses:\n" %i)
		for j in range(0, no_clauses, 2):
			l=[]
			for k in range(no_features):
				if tm.ta_action(i, j, k) == 1:
					if k < int(no_features/2):
						l.append(" x%d" %k)
					else:
						l.append("~x%d" %(k-int(no_features/2)))
			expr_file.write("Clause #%d: \n" %j)
			expr_file.write(" & ".join(l))
			expr_file.write("\n")

		expr_file.write("\nClass %d Negative Clauses:\n" %i)
		for j in range(1, no_clauses, 2):
			l=[]
			for k in range(no_features):
				if tm.ta_action(i, j, k) == 1:
					if k < int(no_features/2):
						l.append(" x%d" %k)
					else:
						l.append("~x%d" %(k-int(no_features/2)))
			expr_file.write("Clause #%d: \n" %j)
			expr_file.write(" & ".join(l))
			expr_file.write("\n")

	expr_file.close()

def ClausePrunedExpr(expr_file_name, tm, pruned):
	actual_index = []
	for i,each in enumerate(pruned):
		if each == False:
			actual_index.append(i)

	no_classes = tm.number_of_classes
	no_clauses = tm.number_of_clauses
	no_features = tm.number_of_features
	expr_file = open(expr_file_name, "w")
	for i in range(no_classes):
		expr_file.write("\nClass %d Positive Clauses:\n" %i)
		for j in range(0, no_clauses, 2):
			l=[]
			for k in range(no_features):
				if tm.ta_action(i, j, k) == 1:
					if k < int(no_features/2):
						l.append(" x%d" %actual_index[k])
					else:
						l.append("~x%d" %actual_index[k-int(no_features/2)])
			expr_file.write("Clause #%d: \n" %j)
			expr_file.write(" & ".join(l))
			expr_file.write("\n")

		expr_file.write("\nClass %d Negative Clauses:\n" %i)
		for j in range(1, no_clauses, 2):
			l=[]
			for k in range(no_features):
				if tm.ta_action(i, j, k) == 1:
					if k < int(no_features/2):
						l.append(" x%d" %actual_index[k])
					else:
						l.append("~x%d" %actual_index[k-int(no_features/2)])
			expr_file.write("Clause #%d: \n" %j)
			expr_file.write(" & ".join(l))
			expr_file.write("\n")

	expr_file.close()

def TA_states(state_file_name, tm):
	state_file = open(state_file_name, "w")
	state_list = tm.get_state()
	for i in range(tm.number_of_classes):
		state_file.write("Class %d: \n[" %i)
		for each in state_list[i][1]:
			state_file.write("%d " %each)
		state_file.write("]\n\n")
	state_file.close()

def FalseXRate(tm, X_set, Y_set):
	all_classes = list(set(Y_set))
	FP = [0] * len(all_classes)
	FN = [0] * len(all_classes)
	Y_pred = tm.predict(X_set) 
	for each_pred, each_act in zip(Y_pred, Y_set):
		if each_pred != each_act:
			FP[all_classes.index(each_pred)] = FP[all_classes.index(each_pred)] + 1
			FN[all_classes.index(each_act)] = FN[all_classes.index(each_act)] + 1
	FP = [format(float(i)/len(Y_set)*100, '.2f') for i in FP]
	FN = [format(float(i)/len(Y_set)*100, '.2f') for i in FN]
	return FP, FN

def PreTrainPrune(X_train_org, X_test_org):
	X_org = np.concatenate((X_train_org, X_test_org), axis=0)
	features_and = []
	features_nor = []
	for i in range(len(X_org[0])):
		features = [row[i] for row in X_org]
		features_and.append(all(features))
		features = [1-row[i] for row in X_org]
		features_nor.append(all(features))
	features_unchanged = []
	for each1, each2 in zip(features_and, features_nor):
		features_unchanged.append(any([each1, each2]))
	X_prune = []
	for i in range(len(X_org)):
		X_prune.append([])
	for i,each in enumerate(features_unchanged):
		if each == False:
			for j in range(len(X_org)):
				X_prune[j].append(X_org[j][i])
	X_train_prune = np.array(X_prune[0:len(X_train_org)])
	X_test_prune = np.array(X_prune[len(X_train_org):])
	return X_train_prune, X_test_prune, features_unchanged

def ClauseSum(tm, X, epoch_no):
	no_classes = tm.number_of_classes
	no_clauses = tm.number_of_clauses
	no_features = tm.number_of_features
	pos_clauses = []
	neg_clauses = []
	for i in range(no_classes):
		# Positive clauses
		for j in range(0, no_clauses, 2):
			pos_clauses.append([])
			for k in range(no_features):
				if tm.ta_action(i,j,k) == 1:
					pos_clauses[i*int(no_clauses/2)+int(j/2)].append(1)
				else:
					pos_clauses[i*int(no_clauses/2)+int(j/2)].append(0)

		# Negative clauses
		for j in range(1, no_clauses, 2):
			neg_clauses.append([])
			for k in range(no_features):
				if tm.ta_action(i,j,k) == 1:
					neg_clauses[i*int(no_clauses/2)+int((j-1)/2)].append(1)
				else:
					neg_clauses[i*int(no_clauses/2)+int((j-1)/2)].append(0)
	
	clause_sums = []
	#clause_sum_file = open("ClauseSum/ClauseSum"+str(epoch_no)+".txt", "w")
	#clause_sum_file.write("[")
	posClause_sum_file = open("ClauseSum/PosClauseSum"+str(epoch_no)+".txt", "w")
	posClause_sum_file.write("[")
	negClause_sum_file = open("ClauseSum/NegClauseSum"+str(epoch_no)+".txt", "w")
	negClause_sum_file.write("[")
	for i,each in enumerate(X):
		print("Computing clause sums for sample %d/%d ..." %(i,len(X)))
		negated_each = [lit^1 for lit in each]
		negated_each = np.array(negated_each)
		all_lit = np.concatenate((each,negated_each), axis=0)
		all_lit = all_lit.tolist()
		# Positive  clause
		pos_clause_outputs = []
		for clause in pos_clauses:
			clause_output = all([all_lit[j] for j,x in enumerate(clause) if x==1])
			pos_clause_outputs.append(clause_output)
		pos_clause_sum = [sum(pos_clause_outputs[j:j+int(no_clauses/2)]) for j in range(0, len(pos_clause_outputs), int(no_clauses/2))]
		# Negative clause
		neg_clause_outputs = []
		for clause in neg_clauses:
			clause_output = all([all_lit[j] for j,x in enumerate(clause) if x==1])
			neg_clause_outputs.append(clause_output)
		neg_clause_sum = [sum(neg_clause_outputs[j:j+int(no_clauses/2)]) for j in range(0, len(neg_clause_outputs), int(no_clauses/2))]
		
		clause_sum = [x-y for x,y in zip(pos_clause_sum,neg_clause_sum)]
		clause_sums.append(clause_sum)

		#for each_clause_sum in clause_sum:
		#	clause_sum_file.write(str(each_clause_sum)+" ")
		#clause_sum_file.write(";\n")
		for PosEach, NegEach in zip(pos_clause_sum, neg_clause_sum):
			posClause_sum_file.write(str(PosEach)+" ")
			negClause_sum_file.write(str(NegEach)+" ")
		posClause_sum_file.write(";\n")
		negClause_sum_file.write(";\n")
	#clause_sum_file.write("];")
	posClause_sum_file.write("];")
	negClause_sum_file.write("];")

	return clause_sums

# conditional probability table
def CPT(X,Y):
	no_features = len(X[0])
	no_classes = len(set(Y))
	#CPs = [[0 for col in range(no_features*2)] for row in range(no_classes)]
	#CPs_class = [0 for col in range(no_classes)]
	CPs = [[0 for col in range(no_features*2)] for row in range(2)]
	CPs_class = [0,0]
	for i in range(len(X)):
		if Y[i] == 0:
			k = 0
		else:
			k = 1
		CPs_class[k] = CPs_class[k] + 1
		#CPs_class[Y[i]] = CPs_class[Y[i]] + 1
		for j in range(len(X[i])):
			if X[i][j] == 1:
				CPs[k][j] = CPs[k][j] + 1
				#CPs[Y[i]][j] = CPs[Y[i]][j] + 1
			else:
				CPs[k][j+no_features] = CPs[k][j+no_features] + 1
				#CPs[Y[i]][j+no_features] = CPs[Y[i]][j+no_features] + 1
	for i in range(len(CPs)):
		for j in range(len(CPs[i])):
			CPs[i][j] = float(CPs[i][j] / CPs_class[i])
	return CPs

# X and Y for verification
def XY4verification(X, Y, tm, XY_file_path):
	X_file = open(XY_file_path + '/X.txt', 'w')
	Y_file = open(XY_file_path + '/Y.txt', 'w')
	Y_file.write("predict\tactual\n")
	predict_Y = tm.predict(X)
	if len(X) != len(Y):
		print('Number of given samples does not match')
	else:
		for i, label in enumerate(Y):
			for each_x in X[i]:
				X_file.write("%d" %each_x)
			X_file.write("\n")
			Y_file.write("%d\t%d\n" %(predict_Y[i], label))
	X_file.close()
	Y_file.close()
