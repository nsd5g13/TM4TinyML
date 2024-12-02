import os

dataset = 'sports'
no_trees = [5,10,15,20,25,30]
max_depth = [2,4,6,8,10,12,14,16,18,20]
#no_trees = [25,30]
#max_depth = [6,8,10,12,14,16,18,20]
runs = 3

open(r'log/results.txt', 'w').close()

for no_tree in no_trees:
	for depth in max_depth:
		print("Number of trees: %d, max depth: %d" %(no_tree, depth))
		script = "python3 main.py %s %s %s" %(dataset, no_tree, depth)
		print(script)
		for run in range(runs):
			os.system(script)
