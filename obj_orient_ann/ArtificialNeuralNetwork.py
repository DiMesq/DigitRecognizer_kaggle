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

			layers_weight = (2 * np.random.randn(layers_sizes[i+1], layers_sizes[i]) 
							* weight_initialize_treshold 
							- weight_initialize_treshold)
			self.weights.append(layers_weight)

	