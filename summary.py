import torch
from nets.yolo import YoloBody
from configure import *
from thop import profile

model1 = YoloBody(num_classes=15, phi='m')

img = torch.rand([1, 3, 640, 640])
flops1, params1 = profile(model1, inputs=(img,))
print('model1 Params = ' + str(params1 / 1000 ** 2) + 'M')
