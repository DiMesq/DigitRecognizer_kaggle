# hyper param and such
REGUL_FACTOR = 0.0001
LEARN_RATE = 0.1
BATCH_SIZE = 30
MAX_EPOCHS = 30

# in data filenames
TRAIN_DATA = '../train.csv'
TEST_DATA = '../test.csv'

# output filenames
MY_TEST_OUTPUT = "predictions.out"
BEST_THETAS = "net_weights.json"

# data info
N_TRAIN_EXAMPLES = 42000
N_TEST_EXAMPLES = 28000
N_PIXELS_PER_IMAGE = 784
MAX_PIXEL_VAL = 255

# layers for the net
LAYERS_SIZES = [N_PIXELS_PER_IMAGE, 15, 15, 10]


