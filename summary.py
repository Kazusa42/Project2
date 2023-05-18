import torch
from torchvision.models._utils import IntermediateLayerGetter
from nets.yolo import YoloBody
from configure import *
from thop import profile

import matplotlib.pyplot as plt
import numpy as np

model = YoloBody(num_classes=15, phi='m')
# model = torchvision.models.resnet18()
for name in model.backbone.backbone.dark2.state_dict():
    print(name)

img = torch.rand([1, 3, 640, 640])
"""flops1, params1 = profile(model, inputs=(img,))
print('model1 Params = ' + str(params1 / 1000 ** 2) + 'M')"""

mid_out = IntermediateLayerGetter(model.backbone.backbone.dark2,
                                  {'1.m.1.mlp2.conv.conv.bias': 'feat2'})
"""out = mid_out(torch.rand(1, 3, 1024, 1024))
print([(k, v.shape) for k, v in out.items()])"""
print(mid_out)

# Draw attention map

# out_features = model.backbone.forward(img)
# print(len(out_features))

# layer3 = torch.sum(out_features[0], dim=1)
# print(layer3.shape)

# layer3 = torch.sum(layer3, dim=0)
# print(layer3.shape)

# create attention heat-map figure
"""fig, axe = plt.subplots(figsize=(layer3.shape[0], layer3.shape[1]))
axe.set_xticks(np.arange(layer3.shape[0]))
axe.set_yticks(np.arange(layer3.shape[1]))
values = layer3.detach().numpy()"""
# print(values)

"""attention_heat_map = axe.imshow(values)
axe.figure.colorbar(attention_heat_map, ax=axe,
                    fractions=1)"""
"""plt.imshow(values)
plt.colorbar(fraction=0.05, pad=0.05)
plt.savefig('attention_heat_map')"""

