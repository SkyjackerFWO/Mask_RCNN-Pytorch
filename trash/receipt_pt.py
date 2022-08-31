"""
Mask R-CNN
Configurations and data loading code for MS Receipt.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained Receipt weights
    python3 Receipt.py train --dataset=/path/to/Receipt/ --model=Receipt

    # Train a new model starting from ImageNet weights
    python3 Receipt.py train --dataset=/path/to/Receipt/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 Receipt.py train --dataset=/path/to/Receipt/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 Receipt.py train --dataset=/path/to/Receipt/ --model=last

    # Run Receipt evaluatoin on the last model you trained
    python3 Receipt.py evaluate --dataset=/path/to/Receipt/ --model=last
"""

import os
import time
import numpy as np
import json
import skimage.draw


# Download and install the Python Receipt tools from https://github.com/waleedka/Receipt
# That's a fork from the original https://github.com/pdollar/Receipt with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/Receiptdataset/Receiptapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
Receipt_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_Receipt.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
##DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################

class ReceiptConfig(Config):
    """Configuration for training on MS Receipt.
    Derives from the base Config class and overrides values specific
    to the Receipt dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Receipt"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # Background + receipt
    
    # Number of trainning steps per epoch
    STEPS_PER_EPOCH = 100
    
    # Skip detection wwith < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ReceiptDataset(utils.Dataset):
    def load_receipt(self, dataset_dir, subset):
        """Load a subset of the Receipt dataset.
        dataset_dir: The root directory of the Receipt dataset.
        subset: What to load (train, val)
        """

        # Add classes
        self.add_classes("reciept",1,"reciept")
        
        # Train or validation dataset?
        assert subset in ["train","val"]
        dataset_dir = os.path.join(data_dir, subset)
        
        # Load data
        # Data from Manh Nguyen have structer
        # {
        # "licenses": [
        #     {
        #         "name": "",
        #         "id": 0,
        #         "url": ""
        #     }
        # ],
        # "info": {
        #     "contributor": "",
        #     "date_created": "",
        #     "description": "",
        #     "url": "",
        #     "version": "",
        #     "year": ""
        # },
        # "categories": [
        #     {
        #         "id": 1,
        #         "name": "Receipt",
        #         "supercategory": ""
        #     }
        # ],
        # "images": [
        #     {
        #         "id": 1,
        #         "width": 768,
        #         "height": 1024,
        #         "file_name": "mcocr_public_145014zzzej.jpg",
        #         "license": 0,
        #         "flickr_url": "",
        #         "coco_url": "",
        #         "date_captured": 0
        #     },...
        # "annotations": [
        #     {
        #         "id": 1,
        #         "image_id": 1,
        #         "category_id": 1,
        #         "segmentation": [
        #             [
        #                47.65,875.4,486.25,871.69,640.5,858.68,638.64,628.23,590.32,275.13,47.65,310.44,21.64,464.69,17.92,661.68,30.93,827.08
        #             ]
        #         ],
        #         "area": 346215,
        #         "bbox": [
        #             17.92,
        #             275.13,
        #             622.58,
        #             600.27
        #         ],
        #         "iscrowd": 0,
        #         "attributes": {
        #             "occluded": false
        #         }
        #     },...
        # }
        
        # We will tranform:
        # {
        #     <image_id>:{
        #         "file_name": "mcocr_public_145014zzzej.jpg",
        #         "width": 768,
        #         "height": 1024,
        #         "segmentation": {[
        #             [
        #                 47.65,875.4,486.25,871.69,640.5,858.68,638.64,628.23,590.32,275.13,47.65,310.44,21.64,464.69,17.92,661.68,30.93,827.08
        #             ],...
        #         ]}
        #     },..
        # }
        
        data =json.load(open(os.path.join(data_dir, "instances_default.json")))
        
        new_data = {}
        for element in data["images"]:
            id = element['id']
            dict = {
                "file_name":element['file_name'],
                "width":element['width'],
                "height": element['height'],
            }
            new_data[id] = dict
    
        for element in data["annotations"]:
            # id = element['id']
            # temp = [file_name[id],element['bbox']]
            # print(temp)
            # temp_arr.append(temp)
            try:
                image_id = element['image_id']
                dict = new_data.get(image_id)
                dict["segmentation"] = element["segmentation"]
                new_data[image_id] = dict
            except:
                print("Can't fine file by id ",id)
                
        for element in new_data:
            image_path = os.path.join(dataset_dir, element['file_name'])
            image = skimage.io.imread(image_path)
            self.add_image(
                    "receipt", 
                    image_id = element['file_name'], # use file name as a unique image id
                    path = image_path,
                    height = element['height'],
                    width = element['width'],
                    polygons = element["segmentation"] 
                )


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Receipt image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Receipt":
            return super(ReceiptDataset, self).load_mask(image_id)

            # # Build mask of shape [height, width, instance_count] and list
            # # of class IDs that correspond to each channel of the mask.
            # for annotation in annotations:
            #     class_id = self.map_source_class_id(
            #         "Receipt.{}".format(annotation['category_id']))
            #     if class_id:
            #         m = self.annToMask(annotation, image_info["height"],
            #                            image_info["width"])
            #         # Some objects are so small that they're less than 1 pixel area
            #         # and end up rounded out. Skip those objects.
            #         if m.max() < 1:
            #             continue
            #         # Is it a crowd? If so, use a negative class ID.
            #         if annotation['iscrowd']:
            #             # Use negative class ID for crowds
            #             class_id *= -1
            #             # For crowd masks, annToMask() sometimes returns a mask
            #             # smaller than the given dimensions. If so, resize it.
            #             if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
            #                 m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
            #         instance_masks.append(m)
            #         class_ids.append(class_id)

            # # Pack instance masks into an array
            # if class_ids:
            #     mask = np.stack(instance_masks, axis=2)
            #     class_ids = np.array(class_ids, dtype=np.int32)
            #     return mask, class_ids
            # else:
            #     # Call super class to return an empty mask
            #     return super(ReceiptDataset, self).load_mask(image_id)
            
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"],len(info["polygons"])],
                         dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            #Get indexes of pixels inside the polygons and set them to 1
            all_points_x = []
            all_points_y = []
            for i in range(len(p)):
                if i % 2 == 0:
                    all_points_x = p[i]
                    all_points_y = p[i]
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr,cc,i] = 1
            
            
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the Receipt Website."""
        info = self.image_info[image_id]
        if info["source"] == "receipt":
             return info["path"]
        else:
            super(ReceiptDataset, self).image_reference(image_id)






############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/Receipt/",
                        help='Directory of the Receipt dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'Receipt'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)

    # Configurations
    if args.command == "train":
        config = ReceiptConfig()
    else:
        class InferenceConfig(ReceiptConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "receipt":
            model_path = Receipt_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = ReceiptDataset()
        dataset_train.load_receipt(args.dataset, "train")
        # dataset_train.load_receipt(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = ReceiptDataset()
        dataset_val.load_dataset_train.load_receipt(args.dataset, "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = ReceiptDataset()
        Receipt = dataset_val.load_receipt(args.dataset, "minival", year=args.year, return_Receipt=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running Receipt evaluation on {} images.".format(args.limit))
        evaluate_Receipt(model, dataset_val, Receipt, "bbox", limit=int(args.limit))
        evaluate_Receipt(model, dataset_val, Receipt, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))