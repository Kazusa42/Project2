"""
Global variable configure
Aiming to simplify the trainging, predicting and evaluating process between different dataset and weights
"""

""" Training setting """

INIT_EPOCH = 55
FREEZE_EPOCH, UNFREZZEZ_EPOCH = 50, 300
FREEZE_BATCH_SIZE, UNFREEZE_BATCH_SIZE = 32, 4


"""  Basic configure for model """

CLASSES_PATH = r'model_data/DOTA_classes.txt'

if INIT_EPOCH == 0:
    MODEL_PATH = r'model_data/yolox_m.pth'
else:
    MODEL_PATH = r'logs/last_epoch_weights.pth'

INPUT_SHAPE = [640, 640]

"""
Proposed methods
1. Attention Neck
2. Multi-scale crop
"""

DATASET_PATH = r'DOTA/Single'

# traditional, none or light
ATTEN_LIST = ['none', 'traditional', 'light']
ATTEN_TYPE = ATTEN_LIST[2]




