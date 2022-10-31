"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset and weights
"""

"""  Basic configure for model """
CLASSES_PATH = r'model_data/DOTA_classes.txt'
MODEL_PATH = r'model_data/yolox_m.pth'

INPUT_SHAPE = [640, 640]

"""
Proposed methods
1. Attention Neck
2. Multi-scale crop
"""
DATASET_PATH = r'DOTA/Single'


""" Training setting """
INIT_EPOCH = 0
FREEZE_EPOCH, UNFREZZEZ_EPOCH = 50, 300
FREEZE_BATCH_SIZE, UNFREEZE_BATCH_SIZE = 64, 8


