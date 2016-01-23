import numpy as np
import helpers as aux

class digit_recognizer_ANN:
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
		self.layers_sizes = [x+1 for lsiz, k in enumerate(layers_sizes) if k != n_layers - 1]
		self.weights = []

		# initialize neural net weights with some random values: -epsilon < value < epsilon
		for i in range(n_layers-1):

			# discard the bias unit from the front layer in the params size 
			# (i.e. decrease the number of lines by 1)
			layers_weight = (2 * np.random.randn(layers_sizes[i+1]-1, layers_sizes[i]) 
							* weight_initialize_treshold 
							- weight_initialize_treshold)
			self.weights.append(layers_weight)

	def train(self, input_pixels, label, learn_rate, regul_factor, batch_size, n_epochs):
		''' input_pixels: (m,n) ndarray, training_examples - must be normalized to 0-1 range
			label: (m,1) ndarray, expected output
			learn_rate: float, gradient descent step size
			regul_factor: float, regularization factor (prevents weights from getting to large)
			batch_size: int, number of examples to take into account in each iteration of gradient descent
			n_epochs: int, number of runs throuh the all of the examples

			return: final cost'''
			
	def _cost_and_gradient(self, input_pixels, label, regul_factor):
		''' inputs: same meaning as in train method
			return: 2 element list, the cost and the gradient (the gradient unrolled)'''


		m = input_pixels.shape[0]
		n = input_pixels.shape[1] + 1 # +1 for the bias unit

		# add the bias unit 
		X = np.ones(m, n)
		X[:, 1:] = input_pixels

		n_layers = len(self.layers_sizes)

		cost = 0

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

				# compute the next layer activations (adds the bias unit at the same time)
				layer_activations = np.ones((self.layer_sizes[l+1], 1))
				layer_activations[1:, :] = h.sigmoid(self.weights(l).dot(nn_activations(l)))

				nn_activations.append(layer_activations)

			# vector of predictions produced by the network
			pred_out = nn_activations[-1]

			# add this example cost
			cost += np.sum(y * log(pred_out) + (1-y) * log(1 - pred_out))

			
			




