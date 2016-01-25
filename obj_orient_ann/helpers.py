import csv
import numpy as np 
import settings as s


def sigmoid(z):
	return 1/(1 + np.exp(-z))

def read_pixels(filedir, n_rows, n_col, has_header, has_labels):
	''' filedir: string, path to csv file. Each element in a row must correspond to the 
				pixels of a single image, where each pixel ranges from 0-255. 
				If this file corresponds to the training data, the labels must be in the 
				first column. The labels range from 0 to 9.

		n_rows: int, number of rows the file has 
				(Important: if file has header, don't include this line in the number of rows)
		n_col: int, number of columns the file has
		has_header: boolean, True if file has header, False otherwise
		has_labels: boolean, True if it has labels according to description in the filedir 
					argument. False otherwise.

		returns: (n_rows, n_col) ndarray, the file content has a numpy array''' 

	# init ndarray to store file data
	data = np.zeros((n_rows, n_col))

	# start reading the file
	with open(filedir, 'rt') as csvfile:
		reader = csv.reader(csvfile)

		# skip header line
		if has_header: next(reader)

		i = 0 #keeps track of which line we are in
		for row in reader:

			# read data
			data[i, :] = [int(ele) for ele in row]

			# normalize pixels to 0-1 range
			if has_labels:
				data[i, 1:] /= s.MAX_PIXEL_VAL
			else: 
				data[i, :] /= s.MAX_PIXEL_VAL

		return data

def split_data(data):
	''' Splits the data into training examples and labels.

		data: ndarray, first column is the labels in 0-9 range. The other 
			  columns for every row have the pixels

		returns : list, of two ndarrays with shapes (m, n) and (m,1) respectively.
				  The first corresponds to the training pixels and the second to
				  the labels -> [training_pixels, labels]'''

	return [data[:, 1:], data[:, 0:1]]

def gradient_check(nn, X, Y, regul_factor):
	''' Checks if the gradient calculation is correct for a given ann.
		Use for debugging purposes

		nn: ArtificialNeuralNetwork class
		returns: boolean, True if the neural nets gradient computation is correct. False otherwise'''

	# get the nets parameters unrolled
	params = nn.get_unrolled_params()

	#compute the numerical aproximation of the gradient
	grad_aprox = gradient_numerical_aproximation(nn, params, X, Y, regul_factor)

	# get the neural network computed gradient
	[c, nn_grad] = nn.cost_and_gradient(X, Y, regul_factor)

	#unroll the net computed gradient
	nn_grad_unrolled = []
	for D in nn_grad:
		nn_grad_unrolled += list(D.flatten())

	nn_grad_unrolled = np.array(nn_grad_unrolled)

	# check if the numerical aproximation and the gradient from backprop are similar
	r = np.linalg.norm(nn_grad_unrolled - grad_aprox) / np.linalg.norm(nn_grad_unrolled + grad_aprox)

	# check if the gradients are similar
	print("result: ", r)
	return r<10**(-5)

def gradient_numerical_aproximation(obj, params, *args):
	''' Computes a numerical aproximation of the gradient of the objective 
		function method of the object obj. Assumes this method is named 
		"cost_and_gradient" and that the object obj has a setter for its 
		parameters called "set_params" 

		*args: arguments to the object's objective function method. '''

	def modify_parameter(params, epsilon, index):
		''' Modifies the input parameter in params[index] by epsilon and returns
			the new parameters vector

			params: list, list of parameters [x1, x2, ... , x_index, ... , xn]
			epsilon: float, small float (positive or negative) -> ~10**(-4) 
			returns: list, list of parameters modified in one input -> 
					  [x1, x2, ... , x_index + epsilon, ..., xn] '''

		params_out = params[:]
		params_out[index] += epsilon
		return params_out

	epsilon = 10**(-4)

	print("this: ", args[2])
	# calculate the aproximation to the gradient
	n = len(params)

	grad_aprox = []
	for i in range(n):
		print("Iter " + str(i) + " of " + str(n))
		params_mod = modify_parameter(params, epsilon, i)
		obj.set_params(params_mod)
		[cost1, g] = obj.cost_and_gradient(*args)

		params_mod = modify_parameter(params, -epsilon, i)
		obj.set_params(params_mod)
		[cost2, g] = obj.cost_and_gradient(*args)

		grad_aprox.append((cost1 - cost2) / (2*epsilon))

	# set the objects params back to original
	obj.set_params(params)

	return np.array(grad_aprox)

	

