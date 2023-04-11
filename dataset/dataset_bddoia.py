"""
The BDD-OIA dataset is from the paper
"Explainable Object-induced Action Decision for Autonomous Vehicles"
The implementation of the dataloader is adapted from the code
https://github.com/Twizwei/bddoia_project
Please site this paper to use the BDD-OIA dataset or the code

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
import json
import random
import transform_bddoia as T

# For bdd-oia dateset
class BddoiaDataset(Dataset):
    def __init__(self, Root, cropSize=(1280, 720), train_set=True):
        super(BddoiaDataset, self).__init__()

        self.gtRoot = os.path.join(Root, "annotations")
        self.imageRoot = os.path.join(Root, "data")
        self.cropSize = cropSize

        if train_set:
            with open(os.path.join(self.gtRoot, "train_25k_images_actions.json")) as json_file:
                action = json.load(json_file)
            with open(os.path.join(self.gtRoot, "train_25k_images_reasons.json")) as json_file:
                reason = json.load(json_file)
        else:
            with open(os.path.join(self.gtRoot, "val_25k_images_actions.json")) as json_file:
                action = json.load(json_file)
            with open(os.path.join(self.gtRoot, "val_25k_images_reasons.json")) as json_file:
                reason = json.load(json_file)

        action['images'] = sorted(action['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = action['annotations']
        imgNames = action['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            # print(len(action_annotations[ind]['category']))
            if len(action_annotations[ind]['category']) == 4 or action_annotations[ind]['category'][4] == 0:
                file_name = osp.join(self.imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        if train_set:
            print("number of samples in dataset:{}".format(len(self.reasons)))
        self.perm = list(range(self.count))
        random.shuffle(self.perm)

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        # test = True
        imgName = self.imgNames[self.perm[ind]]
        target = np.array(self.targets[self.perm[ind]], dtype=np.int64)
        reason = np.array(self.reasons[self.perm[ind]], dtype=np.int64)

        img_ = Image.open(imgName)

        color_jitter = T.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            )
        normalize_transform = T.Normalize(
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.],
                to_bgr255=True,
            )
        transform = T.Compose(
                [color_jitter,
                 #T.Resize(self.cropSize[1], self.cropSize[0]),
                 T.ToTensor(),
                 normalize_transform,
                ]
            )
        img, target = transform(img_, target)
        label_action = torch.FloatTensor(target)[0:4]
        label_reason = torch.FloatTensor(reason)
        labels = [label_action, label_reason]

        return img, labels