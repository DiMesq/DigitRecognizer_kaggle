from helpers import *
from neural_net import *
import numpy as np

N_TRAIN_EXAMPLES = 42000
MAX_POSSIBLE_INPUT = 255

train_set = read_data("train.csv", N_TRAIN_EXAMPLES)

train_X, train_Y = parse_data(train_set, MAX_POSSIBLE_INPUT)

# network with a input layer of 784, hidden layer of 15 and an output layer of 10 neurons
# add the intercept (bias unit) to all the layers (except the output layer)
s1 = 784 + 1
s2 = 15 + 1
s3 = 10

EPSILON = 1 * 10**(-2)
LAMBDA = 0.1
N_LAYERS = 3

# init the layers weights (parameters) as random values between -eps and eps
Theta1 = 2 * EPSILON * np.random.random((s2, s1)) - EPSILON
Theta2 = 2 * EPSILON * np.random.random((s3, s2)) - EPSILON

Matrix_Thetas = [Theta1, Theta2]


# unroll thetas
thetas = np.zeros( sum([x.size for x in Matrix_Thetas]) )
index = 0
for i in range(N_LAYERS - 1):
	Theta = Matrix_Thetas[i]
	siz = Theta.size
	thetas[index : index + siz] = Theta.flatten()

	index += siz

pred = neural_net(thetas, train_X, train_Y, LAMBDA, [s1, s2, s3])

print(pred)



