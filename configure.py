"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset and weights
"""

"""  Basic configure for model """
CLASSES_PATH = r'model_data/DOTA_classes.txt'
MODEL_PATH = r'logs/best_epoch_weights.pth'
IF_CUDA = r'True'
CONFIDENCE = 0.5
NMS_SCORE = 0.3
INPUT_SHAPE = [640, 640]
RESOLUTION = [INPUT_SHAPE[0] // 32, INPUT_SHAPE[1] // 32]

"""
Proposed methods
1. Attention Neck
2. Multi-scale crop
"""
IF_ATTENTION = True

DATASET_PATH = r'DOTA/Multi'

""" Training setting """
PRE_TRAINED = False  # If MODEL_PATH is not None, the value will not work.
MOSAIC = True

INIT_EPOCH = 0
FREEZE_EPOCH, UNFREZZEZ_EPOCH = 50, 300
FREEZE_BATCH_SIZE, UNFREEZE_BATCH_SIZE = 64, 12
FREEZE_TRAIN = True

OPT_TYPE = r'adam'  # sgd or adam
INIT_LR = 0.001  # If you use adam, set this value to 0.001
MIN_LR = INIT_LR * 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
LR_DECAY_TYPE = r'cos'
