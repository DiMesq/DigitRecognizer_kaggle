import helpers as h
import settings as s
import ArtificialNeuralNetwork as ann

nn1 = ann.ArtificialNeuralNetwork(s.LAYERS_SIZES)

# In the number of columns argument: the "+1" is to account for the labels column
data = h.read_pixels(s.TRAIN_DATA, s.N_TRAIN_EXAMPLES, s.N_PIXELS_PER_IMAGE + 1, True, True)

[train_data, train_labels] = h.split_data(data)

nn1.train(train_data, train_labels, s.LEARN_RATE, s.REGUL_FACTOR, s.BATCH_SIZE, s.MAX_EPOCHS)

h.write_json_object(nn1.get_unrolled_params(), s.BEST_THETAS)


