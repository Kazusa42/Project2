# Project1
Code for thesis. Mainly focus on 2 problems.  
```
1. High resolution image.
2. Multi-scale scenario
```

According to these questions, I mainly proposed 3 methods to adress them.
```
1. Multi-scale crop.  
2. Multi-head self-attention.  
```
---

## Multi-scale crop (Dataset)
The original dataset is DOTA v1.0. The original train set will be used as train and val set. The original val set will be treated as test set.

For multi-scale crop training, the original train set is cropped into 2 scales:  
```
subsize = 640, overlap = 50
subsize = 1280, overlap = 100
```

For multi-scale crop evaluateing, the original val set is cropped into 3 scales:
```
subsize = 640, overlap = 50
subsize = 1280, overlap = 100
```

The model is trained and evaluated on `MULTICROPPED_DOTA` dataset.

And in order to evaluate the influence brought by `Multi-scale crop`, the model also trained under `SINGLECROPPED_DOTA` to perform a baseline. The cropping params are listed blow.  
```
Single scale crop, applied on train, val and test set.
subsize = 1024, overlap = 50
```
After cropping the images, the images without objects are removed. This process is quite significant, reduces the dataset size by over 30%.  

About how to processing the data, refer to another project `DOTA-processing`.

---

## Multi-head self-attention (AttentionNeck)  
Use a pure global mulit-head self attention block to replace the depth-wise convolution in bottleneck. More etails about this design is shown blow.

The backbone structure is modified from a convnext_tiny, only the depthwise convolution in stage 4 is replaced by a AttentionNeck.

![AttentionNeck](model_data/multi-scale crop.png "AttentionNeck")

![MHSA](model_data/attention block.png "MHSA")

---

## Usage
The label format is using VOC-style. So, first transform the annotations into `xml` file.

To set the training scheduel, go to `configure.py`, almos every training paramaters are in this script.

Run `train.py` to train the model. If train a model from very begining (do not pre-trained backbone or pre-trained model), set the `epoch` to some big value, such as 500. If use pre-trained backbone or model, `epoch` can be set to some small value to save the time, such as 300.

To evaluate the model, run `get_map.py`. The evaluateing params (such like mAP threshold) are independent from `configure.py`, please directly set them in `get_map.py`.
