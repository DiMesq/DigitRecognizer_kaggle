from helpers import sigmoid
import numpy as np	

global iter_cost
iter_cost = []


def net_cost_and_grad(thetas, *args):
	''' Returns the neural net cost and it's gradient for some parameter theta

		theta: 1D vector, the parameters (unrolled) to the network 
		*args: - X -> m by 784 matrix of values between 0.0 and 1.0
		       - Y -> 1D vector of size m, the expected output for the input x
		       - regula -> float, regularization hyper-parameter
		       - layer_sizes -> 1D vector of ints, the layer sizes by order, 
		       					[input layer, 
		       					 hidden_layer_1, (...), hidden_layer_n, 
		       					 output_layer]'''
	global iter_cost

	X = args[0]
	Y = args[1]
	regula = args[2]
	layer_sizes = args[3]

	m = X.shape[0] 
	n_layers = len(layer_sizes)

	# roll-up parameters (weights)
	list_thetas = []
	list_deltas = []
	k = 0
	for i in range(n_layers - 1):
		l = layer_sizes[i]
		j = layer_sizes[i+1]

		list_thetas.append( np.array(thetas[k : k + l*j]).reshape(j, l))
		list_deltas.append( np.zeros((j, l)))

		k += l*j

	###### foward prop, back prop and compute cost ######
	cost = 0
	

	for i in range(m):

		activations = X[i: i+1].copy().transpose()
		layer_outputs = [activations] 

		# foward prop
		for Theta in list_thetas:

			activations = sigmoid(Theta.dot(activations))
			layer_outputs.append(activations)

		# add this training example cost
		output = layer_outputs[-1]

		y = [0 if k != Y[i] else 1 for k in range(10)] # make the Y(i) vector
		y = np.array(y).reshape(10, 1)

		cost += np.sum( y * np.log(output) + (1 - y) * np.log(1 - output))

		# backprop to compute gradient
		layers_errors = [] # last layer errors first and so on

		layers_errors.append(layer_outputs[-1] - y)

		#back propagate the errors
		for k in range(n_layers - 2, 0, -1):
			theta_T = list_thetas[k].transpose()
			prev_errors = layers_errors[-1]
			sigmoid_deriv = layer_outputs[k] * (1 - layer_outputs[k])

			error = theta_T.dot(prev_errors) * sigmoid_deriv
			layers_errors.append(error)

		for k in range(n_layers - 1):
			error = layers_errors[n_layers - 2 - k]
			activation = layer_outputs[k].transpose()

			list_deltas[k] += error.dot(activation)

	D = []
	for k in range(n_layers - 1):		
		theta = list_thetas[k]
		delta = list_deltas[k]

		Dk = (1/m) * delta 
		Dk[:, 1:] = Dk[:, 1:] + regula * theta[:, 1:]
		D.append(Dk)

	# unroll parameters again
	grad_out = []
	for Dk in D:
		grad_out += list(Dk.flatten().reshape(-1,))

	# add the regularization term to the cost 
	thetas_cost = 0
	for Theta in list_thetas:
		thetas_cost += np.sum(Theta[:, 1:] ** 2) # removes the weights of the bias units

	cost = (-1 / m) * cost
	cost += (regula / (2 * m)) * thetas_cost 

	iter_cost.append(cost)
	return cost, np.array(grad_out)	

def use_net(thetas, layer_sizes, input):
	''' thetas: list of floats, the net's weights
	    n_layers: int, number of layers the neural net has
	    input: ndarray of shape (785, 1), the input example to get the networks output'''

	n_layers = len(layer_sizes)

	# roll-up parameters (weights)
	list_thetas = []
	k = 0
	for i in range(n_layers - 1):
		l = layer_sizes[i]
		j = layer_sizes[i+1]

		list_thetas.append( np.array(thetas[k : k + l*j]).reshape(j, l))

		k += l*j

	activations = input.copy()

	for theta in list_thetas:

		activations = sigmoid(theta.dot(activations))

	max_val = np.amax(activations)

	return [i for i, j in enumerate(activations[:, 0]) if j == max_val][0]








		

