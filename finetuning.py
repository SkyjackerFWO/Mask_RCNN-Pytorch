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


class ReceiptDataset(torch.utils.data.Dataset):
    def __init__(self,root,annFile,catNms = 'Receipt',coco=True,transform=None):
        self.root = root
        self.transform = transform
        if coco == True:
            self.annFile = '{}/{}'.format(root,annFile)
            self.coco = COCO(self.annFile)
            self.catIds = self.coco.getCatIds(catNms=catNms)
            self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        else:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def __getitem__(self, idx):
        num_objs = 1
        
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)

        
        img_path = '{}/{}'.format(self.root,img['file_name'])
        

        # print(img_path)
        img = Image.open(img_path).convert("RGB")
        
        #load mask from ann to bin-map
        anns = self.coco.loadAnns(annIds)
        mask = self.coco.annToMask(anns[0])
        mask = np.expand_dims(mask, axis=0)
        masks = torch.as_tensor(mask, dtype=torch.uint8)
        
        boxes =anns[0]['bbox']
        #print(boxes)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.unsqueeze(0)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        ar = anns[0]
        image_id = torch.tensor([idx])
        area = torch.as_tensor(anns[0]['area'], dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes 
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target
    
    def __len__(self):
        return len(self.imgIds)
    
# dataset = ReceiptDataset('datasets/val',"instances_default.json")
# print(len(dataset))
# for i in range(len(dataset)):
#     image, target = dataset[i]
#     boxes = target['boxes']
#     for box in boxes:
#         # if box[0] >= box[2] or box[1] >= box[3]: # min x > max x or min y > max y
#         print(i, box)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




# ###Testing forward() method
# dataset = ReceiptDataset('datasets/train',"instances_default.json",transform = get_transform(train=True))
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# #dataset = PennFudanDataset('PennFudanPed', transform = get_transform(train=True))
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=2, shuffle=True, num_workers=4,
#     collate_fn=utils.collate_fn
# )
# # For Training
# images,targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]

# for target in targets:
#     print("mask: ",target['masks'].size())
#     print("bbox: ",target['boxes'].size())
# output = model(images,targets)   # Returns losses and detections
# # For inference
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)           # Returns predictions
# print(predictions)
# ###



# use our dataset and defined transformations
dataset = ReceiptDataset('datasets/train',"instances_default.json",transform = get_transform(train=True))
dataset_test = ReceiptDataset('datasets/val',"instances_default.json",transform = get_transform(train=True))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
indices_test = torch.randperm(len(dataset_test)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices_test[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
# let's train it for 10 epochs
from torch.optim.lr_scheduler import StepLR
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # print('train ok')

    # # update the learning rate
    lr_scheduler.step()
    # print('lr ok')

    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)# pick one image from the test set
    # print('ev ok')

# img, _ = dataset_test[0]
# # put the model in evaluation mode
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])