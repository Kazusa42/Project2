import torch
import torchvision
from nets.yolo import YoloBody
from configure import *
from thop import profile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use('Agg')

"""flops1, params1 = profile(model, inputs=(img,))
print('model1 Params = ' + str(params1 / 1000 ** 2) + 'M')"""

model = YoloBody(num_classes=15, phi='m')
# img = torch.rand([1, 3, 640, 640])
img_path = r'./imgs/test.jpg'

tmp = Image.open(img_path)
trans = torchvision.transforms.ToTensor()
img = trans(tmp).unsqueeze(0)
print(img.shape)

""" Draw attention map """

out_features = model.backbone.backbone.forward(img)
print(len(out_features))

layer3 = torch.sum(out_features['dark2'], dim=1)
# print(layer3.shape)

layer3_flatten = torch.sum(layer3, dim=0)
# print(layer3.shape)

# create attention heat-map figure
fig, axe = plt.subplots(figsize=(layer3_flatten.shape[0], layer3_flatten.shape[1]))
axe.set_xticks(np.arange(layer3_flatten.shape[0]))
axe.set_yticks(np.arange(layer3_flatten.shape[1]))
values = layer3_flatten.detach().numpy()

# attention_heat_map = axe.imshow(values)

plt.imshow(values)
plt.savefig('attention_heat_map')