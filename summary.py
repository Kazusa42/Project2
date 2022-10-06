import torch
from nets.yolo import YoloBody
from configure import *
from thop import profile

"""model = YoloBody(anchors_mask=ANCHOR_MASK, num_classes=15)
x = torch.randn([1, 768, 20, 20])

img = torch.randn([1, 3, 640, 640])
x2, x1, x0 = model.backbone(img)
print(x0.shape)
tmp = model.conv1(x0)
print(tmp.shape)"""

model1 = YoloBody(num_classes=15, phi='m')

img = torch.rand([1, 3, 640, 640])
flops1, params1 = profile(model1, inputs=(img,))
print('model1 Params = ' + str(params1 / 1000 ** 2) + 'M')
