from helpers import *

N_TRAIN_EXAMPLES = 42000

train_set = read_data("train.csv", N_TRAIN_EXAMPLES)

train_X = train_set[:, 1:]
train_Y = train_set[:, 0]


