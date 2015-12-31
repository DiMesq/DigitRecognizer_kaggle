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
s2 = 15 + 1
s3 = 10
layers = [s1, s2, s3]

EPSILON = 1 * 10**(-2)
LAMBDA = 0.001
N_LAYERS = 3

# init the layers weights (parameters) as random values between -eps and eps
Theta1 = 2 * EPSILON * np.random.random((s2, s1)) - EPSILON
Theta2 = 2 * EPSILON * np.random.random((s3, s2)) - EPSILON

Matrix_Thetas = [Theta1, Theta2]

# unroll thetas
thetas = []
for i in range(N_LAYERS - 1):
	thetas += list(Matrix_Thetas[i].flatten().reshape(-1,))

optimize_theta = sp.minimize(nn.net_cost_and_grad, 
							 np.array(thetas), 
							 (train_X, train_Y, LAMBDA, layers), 
							 method='L-BFGS-B', 
							 jac=True, 
							 options={'disp':True})

print(optimize_theta)

# write the best parameters found to a file
if (optimize_theta.success): h.write_json_object(list(optimize_theta.x), s.BEST_THETAS)

# plot the evolution of the cost function with each iteration
plt.figure(1)
plt.plot(nn.iter_cost)
plt.show()









