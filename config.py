
NUM_TRAIN = 50000  # N
NUM_VAL = 50000 - NUM_TRAIN
BATCH = 128  # B
SUBSET = 10000  # M
ADDENDUM = 2500  # K
INITC = 5000

MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda

TRIALS = 5
CYCLES = 7

EPOCH = 200
LR = 0.1
MILESTONES = [160]
EPOCHL = 120  # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4

DATASET = 'cifar10'

NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 32
EMBEDDING_DIM = 256
