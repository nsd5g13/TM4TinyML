import time
import machine
import gc

machine.freq(200000000) # set clock frequency as 200MHz

# load test samples
datapoint_file = 'RF/sports/X.txt'
f = open(datapoint_file)
datapoints = f.readlines()
datapoints = datapoints[0:15] # number of test samples

W_datapoints = []
for datapoint in datapoints:
	datapoint = datapoint.replace('\n', '')
	feature_values = datapoint.split()
	feature_values = [float(x) for x in feature_values]
	W_datapoints.append(feature_values)
f.close()

# load RF model
RF_file = 'RF/sports/RF.txt'
f = open(RF_file)
model_lines = f.readlines()
f.close()

model = [[] for i in range(len(model_lines))]
cnt = 0
for line in model_lines:
	if line != '\n':
		nodes = line.split()
		model[cnt].extend(nodes)
	cnt = cnt + 1

gc.collect()

total_time = 0
# inference
for datapoint in W_datapoints:
	start_time = time.time_ns()

	node_idx = 0
	current_tree_done = False
	all_class = []
	for nodes in model:
		if current_tree_done == False:
			if nodes[node_idx+1] != 'label':
				feature_idx = int(nodes[node_idx])
				split_value = float(nodes[node_idx+1])
				left_idx = int(nodes[node_idx+2])
				right_idx = int(nodes[node_idx+3])

				feature_value = datapoint[feature_idx]
				if feature_value < split_value:
					node_idx = left_idx
				else:
					node_idx = right_idx
	
			else:
				current_class = int(nodes[node_idx])
				all_class.append(current_class)
				current_tree_done = True

		else:
			if len(nodes) == 0:
				current_tree_done = False
				node_idx = 0

	majority = None
	majority_count = 0
	for vote in all_class:
		vote_count = 0
		for other_vote in all_class:
			if vote == other_vote:
				vote_count += 1

		if vote_count > majority_count:
			majority = vote
			majority_count = vote_count

	end_time = time.time_ns()
	total_time = end_time - start_time + total_time

	print("In %.2f milliseconds, predicted class:" %((end_time-start_time)/1000000))
	print(majority)
	gc.collect()

print("Allocated memory: %d Byte" %gc.mem_alloc())
print("Total inference time: %.2f milliseconds" %(total_time/1000000))