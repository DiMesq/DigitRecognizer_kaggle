from helpers import sigmoid
import numpy as np

def neural_net(thetas, *args):
	''' Returns the neural net cost and it's gradient for some parameter theta

		theta: 1D vector, the parameters (unrolled) to the network 
		*args: - X -> m by 784 matrix of values between 0.0 and 1.0
		       - Y -> 1D vector of size m, the expected output for the input x
		       - regula -> float, regularization hyper-parameter
		       - layer_sizes -> 1D vector of ints, the layer sizes by order, 
		       					[input layer, 
		       					 hidden_layer_1, (...), hidden_layer_n, 
		       					 output_layer]'''

	X = args[0]
	Y = args[1]
	regula = args[2]
	layer_sizes = args[3]

	m = X.shape[0] 
	n_layers = len(layer_sizes)

	# roll-up parameters (weights)
	list_thetas = []
	k = 0
	for i in range(n_layers - 1):
		l = layer_sizes[i]
		j = layer_sizes[i+1]

		list_thetas.append( np.array(thetas[k : k + l*j]).reshape(j, l))

		k += l*j

	###### foward prop, back prop and compute cost ######
	cost = 0
	

	for i in range(m):

		layer_outputs = [] 

		activations = X[i].copy().transpose()

		# foward prop
		for Theta in list_thetas:

			activations = sigmoid(Theta.dot(activations))
			layer_outputs.append(activations)

		# add this training example cost
		output = layer_outputs[-1]
		cost += np.sum( Y[i] * np.log(output) + (1 - Y[i]) * np.log(1 - output))


	# add the regularization term to the cost 
	thetas_cost = 0
	for Theta in list_thetas:
		thetas_cost += np.sum(Theta[:, 1:] ** 2) # removes the weights of the bias units

	cost += (regula / (2 * m)) * thetas_cost 

	return cost



	
		
		

