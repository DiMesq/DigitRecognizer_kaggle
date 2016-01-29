import json
import helpers as h
import settings as s
import ArtificialNeuralNetwork
import numpy as np

test_examples = h.read_pixels(s.TEST_DATA, s.N_TEST_EXAMPLES, s.N_PIXELS_PER_IMAGE, True, False)

nn2 = ArtificialNeuralNetwork.ArtificialNeuralNetwork(s.LAYERS_SIZES)

nn2.set_params(h.read_json_object(s.BEST_THETAS))

predictions = []
for i in range(test_examples.shape[0]):

	example = test_examples[i:i+1, :].transpose()

	net_output = nn2.predict(example)

	digit = [k for k in range(len(net_output)) if k == np.argmax(net_output)]

	predictions += digit

h.write_predictions(predictions, s.MY_TEST_OUTPUT)
