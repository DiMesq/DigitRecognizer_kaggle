def neural_net(thetas, *args):
	''' Returns the neural net cost and it's gradient for some parameter theta

		theta: 1D vector, the parameters (unrolled) to the network 
		*args: - X -> m by 784 matrix of values between 0.0 and 1.0
		       - Y -> 1D vector of size m, the expected output for the input x
		       - alpha -> float, learning rate
		       - regula -> float, regularization hyper-parameter
		       - layer_sizes -> 1D vector of ints, the layer sizes by order, 
		       					[input layer, 
		       					 hidden_layer_1, (...), hidden_layer_n, 
		       					 output_layer]'''

	X = args[0]
	Y = args[1]
	alpha = args[2]
	regula = args[3]
	layer_sizes = args[4]
	n_layers = len(layer_sizes)

	# roll-up parameters (weights)
	list_thetas = []
	k = 0
	for i in range(n_layers - 1):
		l = layer_sizes[i]
		j = layer_sizes[i+1]

		list_thetas.append( np.array(thetas[k : k + l*j]).reshape(j, l))

		k += l*j

	
		
		

