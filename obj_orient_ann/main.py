import helpers as h
import settings as s
import DigitRecognizerANN as ann

layers_sizes = [s.N_PIXELS_PER_IMAGE, 5, 10]

nn1 = ann.DigitRecognizerANN(layers_sizes)

# In the number of columns argument: the "+1" is to account for the labels column
data = h.read_pixels(s.TRAIN_DATA, s.N_TRAIN_EXAMPLES, s.N_PIXELS_PER_IMAGE + 1, True, True)

[train_data, train_labels] = h.split_data(data)

print(h.gradient_check(nn1, train_data[13500:13600,:], train_labels[13500:13600,:], s.REGUL_FACTOR))


