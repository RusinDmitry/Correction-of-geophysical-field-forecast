from correction.models.constantBias import ConstantBias
from correction.config import cfg

model = ConstantBias(3).to(cfg.GLOBAL.DEVICE)
print(model)