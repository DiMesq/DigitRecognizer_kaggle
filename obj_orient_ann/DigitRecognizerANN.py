import numpy as np
import helpers as aux

class DigitRecognizerANN:
	''' ANN to recognize hand written digits

		Uses a cross-entropy error cost function

		Note: m is used to refer to the number of examples and n to the number of features (= number of pixels)'''

	def __init__(self, layers_sizes, epsilon):
		''' layers_sizes: list, with each element being one layer size by order.
						  For e.g. a nn with 700 input layer; a hidden layer 
						  with 20 nodes; an 10 node output layer, would be [700, 20, 10]
						  Note: last layer must be of 10 nodes to match the number of digits
			epsilon: used to define the range of possible random values to initialize the nn. 
					 Concretely the init values belong to [-epsilon; +epsilon] '''

		n_layers = len(layers_sizes)

		# add a node to every layer except the output layer - this will be used for the bias unit
		self.layers_sizes = [lsiz+1 if k != n_layers - 1 else lsiz for k, lsiz in enumerate(layers_sizes)]
		self.weights = []
		self.n_params = 0

		# initialize neural net weights with some random values: -epsilon < value < epsilon
		for i in range(n_layers-1):

			# discard the bias unit from the front layer in the params size 
			# (i.e. decrease the number of lines by 1) -> except for the last layer (!)
			next_layer_size = (self.layers_sizes[i+1] if i == n_layers - 2 else 
							   self.layers_sizes[i+1] - 1)  
			layers_weight = (2 * np.random.randn(next_layer_size, self.layers_sizes[i]) 
							* epsilon 
							- epsilon)

			self.weights.append(layers_weight)
			self.n_params += next_layer_size * self.layers_sizes[i]


	def train(self, input_pixels, label, learn_rate, regul_factor, batch_size, n_epochs):
		''' input_pixels: (m,n) ndarray, training_examples - must be normalized to 0-1 range
			label: (m,1) ndarray, expected output
			learn_rate: float, gradient descent step size
			regul_factor: float, regularization factor (prevents weights from getting to large)
			batch_size: int, number of examples to take into account in each iteration of gradient descent
			n_epochs: int, number of runs throuh the all of the examples

			return: final cost'''

		return self.cost_and_gradient(input_pixels, label, regul_factor)

	def cost_and_gradient(self, input_pixels, label, regul_factor):
		''' inputs: same meaning as in train method
			return: 2 element list, the cost and the gradient'''


		m = input_pixels.shape[0]
		n = input_pixels.shape[1] + 1 # +1 for the bias unit

		# add the bias unit 
		X = np.ones((m, n))
		X[:, 1:] = input_pixels

		n_layers = len(self.layers_sizes)

		cost = 0

		# init the list of arrays that will correspond to changes to the weights
		deltas = [np.zeros(Theta.shape) for Theta in self.weights]

		# go over all input examples
		for i in range(m):

			# get a specific example
			Xi = X[i:i+1, :].transpose()

			# get this example's expected vector output (1 in the place of the expected digit and 0 elsewhere)
			y = np.array([1 if k == label[i] else 0 for k in range(10)]).reshape(10, 1)

			# initialize list to store each layer's activation
			nn_activations = [Xi]

			# forward prop
			for l in range(n_layers - 1):

				# compute the next layer activations (adds the bias unit at the same time 
				# -> except for the last layer (!))
				layer_activations = np.ones((self.layers_sizes[l+1], 1))
				
				if l == n_layers - 2:
					layer_activations = aux.sigmoid(self.weights[l].dot(nn_activations[l]))
				else:
					layer_activations[1:, :] = aux.sigmoid(self.weights[l].dot(nn_activations[l]))

				nn_activations.append(layer_activations)

			# vector of predictions produced by the network
			pred_out = nn_activations[-1]

			# add this example cost
			cost += np.sum(y * np.log(pred_out) + (1-y) * np.log(1 - pred_out))

			prev_layer_errors = pred_out - y
			nn_layer_errors = [prev_layer_errors]

			# compute the necessary changes for the last weights (the last Theta)
			deltas[-1] += prev_layer_errors.dot(nn_activations[-2].transpose())

			# backprop
			for l in range(n_layers - 2, 0, -1):

				# back propagate the errors 
				Theta = self.weights[l] 

				# discount the bias unit from the activations when backprop errors
				this_layer_activations = nn_activations[l][1:, :] 
				sigmoid_derivative = this_layer_activations * (1 - this_layer_activations)

				# discounts the bias unit from Theta when back propagating 
				layer_errors = Theta[:,1:].transpose().dot(prev_layer_errors) * sigmoid_derivative
				
				prev_layer_errors = layer_errors
				nn_layer_errors = [layer_errors] + nn_layer_errors

				# compute the necessary changes in the weights
				prev_layer_activations = nn_activations[l-1]
				deltas[l-1] += prev_layer_errors.dot(prev_layer_activations.transpose())

		# compute the final gradient
		gradient = [(1/m) * Delta for Delta in deltas]
		# add regularization
		gradient = [gradient[i] + (regul_factor/m) * self.weights[i] for i in range(n_layers-1)]

		# add regularization to the cost
		cost += (regul_factor / (2*m)) * sum([np.sum(Theta**2) for Theta in self.weights])

		return cost, gradient

	def get_params(self):
		return self.weights

	def get_unrolled_params(self):
		params = []

		for Theta in self.weights:
			params += list(Theta.flatten())

		return params

	def set_params(self, params):
		''' Sets the nets params.

			params: list, unrolled params'''

		if self.n_params != len(params): 
			print("Warning: params not set because number of params introduced"
				   "doesn't match the nets number of params")
			return 

		self.weights = []
		n_layers = len(self.layers_sizes)

		position = 0
		for l in range(n_layers - 1):

			this_layer_size = self.layers_sizes[l]
			next_layer_size = self.layers_sizes[l+1] - 1 #discount bias unit

			if l == n_layers - 2: next_layer_size += 1 # dont discount the bias unit from last layer

			relevant_params = np.array(params[position : position + this_layer_size * next_layer_size])\
								.reshape(next_layer_size, this_layer_size)

			self.weights.append(relevant_params)

			position += this_layer_size * next_layer_size

			




