from helpers import *

N_TRAIN_EXAMPLES = 42000

train_set = read_data("train.csv", N_TRAIN_EXAMPLES)

train_X, train_Y = parse_data(train_set)

# network with a input layer of 784, hidden layer of 15 and an output layer of 10 neurons
s1 = 784
s2 = 15
s3 = 10

EPSILON = 1 * 10**(-2)

# init the layers weights (parameters) as random values between -eps and eps
Theta1 = 2 * EPSILON * np.random.random((s2, s1)) - EPSILON
Theta2 = 2 * EPSILON * np.random.random((s3, s2)) - EPSILON





