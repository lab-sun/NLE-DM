import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from dataset import transform_bdd as T


class BDD_AD(Dataset):
    def __init__(self, Root, cropSize=(1280, 720), train_set=True):
        super(BDD_AD, self).__init__()

        self.root = Root
        self.cropSize = cropSize

        if train_set:
            self.imgroot = os.path.join(self.root, "train_5000")
            with open(os.path.join(self.root, "train_action.json")) as json_file:
                action = json.load(json_file)
            with open(os.path.join(self.root, "train_des.json")) as json_file:
                des = json.load(json_file)
        else:
            self.imgroot = os.path.join(self.root, 'val_2500')
            with open(os.path.join(self.root, "val_action.json")) as json_file:
                action = json.load(json_file)
            with open(os.path.join(self.root, "val_des.json")) as json_file:
                des = json.load(json_file)

        self.imgNames, self.actions, self.desps = [], [], []

        for i in action:
            self.actions.append(i['label'])
            # img_path = os.path.join(self.imgroot, i['file_name'])
            # self.imgNames.append(img_path)
        for i in des:
            self.desps.append(i['label'])
            img_path = os.path.join(self.imgroot, i['file_name'])
            self.imgNames.append(img_path)

        self.count = len(self.imgNames)
        if train_set:
            print("number of samples in dataset:{}".format(len(self.actions)))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        # test = True
        imgName = self.imgNames[ind]
        act = np.array(self.actions[ind], dtype=np.int64)
        des = np.array(self.desps[ind], dtype=np.int64)

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
                 # T.Resize(self.cropSize[1], self.cropSize[0]),
                 T.ToTensor(),
                 normalize_transform,
                ]
            )
        img, act = transform(img_, act)
        label_action = torch.FloatTensor(act)
        label_des = torch.FloatTensor(des)
        labels = [label_action, label_des]

        return img, labels
