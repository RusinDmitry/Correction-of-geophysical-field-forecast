from correction.helpers.ordered_easydict import OrderedEasyDict as edict
import os
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__C.GLOBAL.BATCH_SIZE = None

__C.GLOBAL.BASE_DIR = 'C:/Users/AI Lab/Desktop/Хакатон Москва/Correction-of-geophysical-field-forecast'  # Base project folder

__C.GLOBAL.MODEL_SAVE_DIR = os.path.join(__C.GLOBAL.BASE_DIR, 'logs')
assert __C.GLOBAL.MODEL_SAVE_DIR is not None


__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# __C.GLOBAL.USE_SPATIOTEMPORAL_ENCODING = False
