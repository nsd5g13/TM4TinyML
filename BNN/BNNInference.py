import math, time, gc
import machine

machine.freq(200000000) # set clock frequency as 200MHz

# define a single dense layer followed by activation
def dense(nunit, x, w, activation, InputLayer):  
	res = []
	for i in range(nunit):
		z = neuron(x, w[i], activation, InputLayer)
		# print(z)
		res.append(z)
	return res

# perform operation on a single neuron and return a 1d array
def neuron(x, w, activation, InputLayer): 
	tmp = zeros1d(x)
	for i in range(len(x[0])):
		if InputLayer == False:
			tmp = add1d(tmp, [int(str(w[i]) == str(x[j][i])) for j in range(len(x))])
			zero_value = len(x[0])//2
		else:
			if w[i] == '0':
				W = -1
			else:
				W = 1
			tmp = add1d(tmp, [int(x[j][i])*W for j in range(len(x))])
			zero_value = 0

	if activation == "ste_sign":
		yp = ste_sign(tmp, zero_value)
	elif activation == "none":
		yp = tmp
	else:
		print("Invalid activation function--->")
	
	return yp

def zeros(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M

def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed
        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0], list):
        M = [M]

    # Section 2: Get dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros(cols, rows)

    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT

# Step sign activation function
def ste_sign(x, zero_value):	
	# print(x)
	y = []
	for i in range(len(x)):
		if x[i] >= zero_value:
			y.append(1)
		else:
			y.append(0)
	# print(y)
	return y

# Softmax activation function
def softmax(x):
	temp = [math.exp(v) for v in x]
	total = sum(temp)
	return [t / total for t in temp]

# Extract binary weights from given text file
def read_weights(weights_file_path):
	f = open(weights_file_path, 'r')
	weight_lines = f.readlines()
	f.close()
	weights = []
	for each in weight_lines:
		 weights.append(each.replace('\n', ''))
	return weights

# Export test samples from given text file
def read_samples(samples_file_path):
	f = open(samples_file_path, 'r')
	sample_lines = f.readlines()
	f.close()
	samples = []
	for each in sample_lines:
		 samples.append(each.replace('\n', ''))
	return samples

def zeros1d(x):  # 1d zero matrix
	z = [0 for i in range(len(x))]
	return z

def add1d(x, y):
	if len(x) != len(y):
		print("Dimention mismatch")
		exit()
	else:
		z = [x[i] + y[i] for i in range(len(x))]
	return z	

# ------------------------ main ------------------------------------
all_samples = read_samples(r'redress/statlog_bnn/X.txt')
weights0 = read_weights(r'redress/statlog_bnn/weights0.txt')
weights1 = read_weights(r'redress/statlog_bnn/weights1.txt')
gc.collect()
W = 20
no_iterations = len(all_samples[0:40])//W
for i in range(no_iterations):
	start_time = time.time_ns()
	samples = all_samples[i*W:(i+1)*W]
	yout1 = dense(len(weights0), samples, weights0, 'ste_sign', True)
	yout1 = transpose(yout1)
	yout2 = dense(len(weights1), yout1, weights1, 'none', False)
	yout2 = transpose(yout2)
	labels = []
	for each_sample in yout2:
		label = each_sample.index(max(each_sample))
		labels.append(label)
	end_time = time.time_ns()
	print("In %.2f seconds, predicted class:" %((end_time-start_time)/1000000000))
	print(labels)
	gc.collect()

print("Allocated memory: %d Byte" %gc.mem_alloc())

