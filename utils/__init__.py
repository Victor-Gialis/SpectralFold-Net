# Configuration parameters for pretraining a model
PRETRAIN_DATASET = "CWRU"
WINDOW_SIZE = 2048
WINDOW_STRIDE = 256
DOWNSAMPLING_FACTOR = 2

# Model parameters
PATCH_SIZE = 16
HIDDEN_DIM = 512
HEADS = 8
N_LAYERS = 3
DROPOUT = 0.2565
MASK_RATIO = 0.25

# Training parameters
EPOCHS = 2
BATCH_SIZE = 64
WEIGHT_DECAY = 1.1133e-5
LEARNING_RATE = 0.0003695

class ArgsPretrain :
    def __init__(self):
        # Dataset parameters
        self.pretrain_dataset = PRETRAIN_DATASET
        self.window_size = WINDOW_SIZE
        self.window_stride = WINDOW_STRIDE
        self.downsampling_factor = DOWNSAMPLING_FACTOR # specific SAP
        # Model parameters
        self.patch_size = PATCH_SIZE
        self.hidden_dim = HIDDEN_DIM
        self.heads = HEADS
        self.n_layers = N_LAYERS
        self.dropout = DROPOUT
        self.mask_ratio = MASK_RATIO # specific MAE
        # Training parameters
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.weight_decay = WEIGHT_DECAY
        self.learning_rate = LEARNING_RATE
class ArgsDownstream :
    def __init__(self):
        # Training parameters
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.weight_decay = WEIGHT_DECAY
        self.learning_rate = LEARNING_RATE

        self.pretrain_dataset = None
        self.finetune_dataset = None
