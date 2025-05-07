
PROBLEM_SIZE = 100  # number of items

# Hyper-Parameters
TOTAL_EPOCH = 200

TRAIN_DATASET_SIZE = 100*1000
TEST_DATASET_SIZE = 100*100
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
# Value-to-Weight Ratio Bonus Factor (Set to 0 to disable)
RATIO_BONUS_FACTOR = 1 # Example value, tune as needed

EMBEDDING_DIM = 128
KEY_DIM = 16  # Length of q, k, v of EACH attention head
HEAD_NUM = 8
ENCODER_LAYER_NUM = 6
FF_HIDDEN_DIM = 512
LOGIT_CLIPPING = 10  # (C in the paper)


ACTOR_LEARNING_RATE = 1e-4
ACTOR_WEIGHT_DECAY = 1e-6

LR_DECAY_EPOCH = 1
LR_DECAY_GAMMA = 1.00


# Logging
LOG_PERIOD_SEC = 15
