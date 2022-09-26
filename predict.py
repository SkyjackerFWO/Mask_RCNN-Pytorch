import os 
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import config as config

img = Image(config.image_path)
# put the model in evaluation mode
model.load_state_dict(torch.load('./checkpoint/ckpt.pth'))
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])