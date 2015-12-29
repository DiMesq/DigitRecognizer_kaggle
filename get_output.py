import helpers as h
import neural_net as nn
import settings as s
import numpy as np

thetas = h.read_json_object(s.BEST_THETAS)

train_set = h.read_pixel_data(s.TRAIN_DATA, s.N_TRAIN_EXAMPLES, False)
test_set = h.read_pixel_data(s.TEST_DATA, s.N_TEST_EXAMPLES, True)

train_X, train_Y = h.parse_data(train_set, s.MAX_POSSIBLE_INPUT)
test_X, test_Y = h.parse_data(test_set, s.MAX_POSSIBLE_INPUT)

s1 = 784 + 1
s2 = 15 + 1
s3 = 10
layers = [s1, s2, s3]

train_res = []

for i in range(train_X.shape[0]):

	train_res.append(nn.use_net(thetas, layers, train_X[i:i+1, :].transpose()))

correct_vector = [1 if train_res[i] == correct_value else 0 for i, correct_value in enumerate(train_Y)]

print("Accuracy on training data: ", np.average(correct_vector))

test_res = []
for i in range(test_X.shape[0]):

	test_res.append(nn.use_net(thetas, layers, test_X[i:i+1, :].transpose()))

h.write_predictions(test_res, s.TEST_OUTPUT)

