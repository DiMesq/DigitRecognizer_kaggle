import helpers as h
import neural_net as nn
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import settings as s

train_set = h.read_pixel_data(s.TRAIN_DATA, s.N_TRAIN_EXAMPLES, False)

train_X, train_Y = h.parse_data(train_set, s.MAX_POSSIBLE_INPUT)

# network with a input layer of 784, hidden layer of 40 and an output layer of 10 neurons
# add the intercept (bias unit) to all the layers (except the output layer)
s1 = 784 + 1
s2 = 100 + 1
s3 = 10
layers = [s1, s2, s3]

EPSILON = 1 * 10**(-4)
LAMBDA = 0.00001
N_LAYERS = 3

# init the layers weights (parameters) as random values between -eps and eps
Theta1 = 2 * EPSILON * np.random.random((s2, s1)) - EPSILON
Theta2 = 2 * EPSILON * np.random.random((s3, s2)) - EPSILON

Matrix_Thetas = [Theta1, Theta2]

# unroll thetas
thetas = []
for i in range(N_LAYERS - 1):
	thetas += list(Matrix_Thetas[i].flatten())

optimize_theta = h.gradient_descent(nn.net_cost_and_grad, 
							 np.array(thetas),
							 0.1,
							 30,
							 20, 
							 train_X, train_Y, LAMBDA, layers)

# write the best parameters found to a file
h.write_json_object(list(optimize_theta), s.BEST_THETAS)


b = h.read_json_object(s.BEST_THETAS)

test_set = h.read_pixel_data(s.TEST_DATA, s.N_TEST_EXAMPLES, True)

test_X, test_Y = h.parse_data(test_set, s.MAX_POSSIBLE_INPUT)
predict = []

for i in range(test_X.shape[0]):
	predict.append(nn.use_net(b, layers, test_X[i:i+1, :].transpose()))

h.write_predictions(predict, s.MY_TEST_OUTPUT)








