import os
import sys

no_clauses = int(sys.argv[1])
T = int(sys.argv[2])
s = float(sys.argv[3])
no_epochs = int(sys.argv[4])
prune_every_no_epochs = int(sys.argv[5])
budget = int(sys.argv[6])
dataset = sys.argv[7]

first_train_epochs=prune_every_no_epochs

open(r'log/AccuracyVsIncludes.txt', 'w').close()

script1 = "python3 StandardTraining.py " + " ".join([str(no_clauses), str(T), str(s), str(first_train_epochs), str(budget), dataset])
script2 = "python3 DIET_Training.py " + " ".join([str(no_clauses), str(T), str(s), str(prune_every_no_epochs), str(budget), dataset])
os.system(script1)

for i in range(1,int(no_epochs/prune_every_no_epochs)):
	os.system(script2+" "+str(i))
