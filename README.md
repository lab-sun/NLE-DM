# NLE-DM
The official pytorch implementation of NLE-DM (Natural-Language Explanations for Decision Making).

![image](images/network.png)

## Usage
* Clone this repo and prepare the environment.
```
git clone https://github.com/lab-sun/NLE-DM.git
cd NLE_DM
conda env create -f environment.yml --name NLE_DM
conda activate NLE_DM
```

* Download the dataset, create the foler `Data` and release it into the `Data`;
```
BDD10K for the pre-training and obtain the semantic segmentation of road scene
BDD_OIA for the Act-Rea sub-network (jointly predict actions and reasons)
BDD_AD for the Act-Desc sub-network  (jointly predict actions and descriptions)
```
* Download the pretrained weight, create the foler `weight` and put into the `weight` (optional);

* To train the network, select the appropriate .py in the folder of `train`
```
pre_train.py: To pretrain the network.
train_act_exp.py: To train the Act-Rea sub-network
train_act_des.py: To train the Act-Desc sub-network
```
* To obatin the prediction results, select the appropriate .py in the folder of `predict`
```
predict_act_rea.py: To jointly predict the driving actions and correpsonding reasons.
predict_act_desc.py: To jointly predict the driving actions and environment descriptions.
```

## Dataset (Download)
Download the datasets and then extract it in the folder of `Data`
* For BDD_AD dataset, please refer to: https: https://drive.google.com/file/d/1NIxloPvfsr4RvN2WjWg96jyd87PUC8e4/view?usp=drive_link
* For BDD100K dataset, please refer to: https://www.bdd100k.com/
* For BDD_OIA dataset, please refer to: https://github.com/Twizwei/bddoia_project/blob/master/README.md

## Pretrained weights (Download)
* Download the pretrained weights and then extract it in the file of `weight`
* The link for pretrained weights is: https://drive.google.com/file/d/1yi-MLL2ry96ChS1hrVp61biEBYxkEW2p/view?usp=drive_link
* The introduction for each weights is as follows:
```
bdd10k_resnet50_1.pth: weight of pre-training on BDD10K
act_rea_resnet50.pth: weight of Act-Rea sub-network
act_des_resnet50.pth: weight of Act-Desc sub-network (backbone: ResNet 50)
```

## Note
* Normally, the backbone is first pre-trained on BDD10K (using pre_train.py), 
then load the pre-trained weight to train the network to jointly predict actions and natural-language reasons/descriptions. 
* To accelerate training process for pre-train on BDD10k dataset, 
we recommend you to load the pre-trained deeplabv3 weights on COCO dataset.
To download the weights, please refer to: 
deeplabv3_resnet50: https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth
deeplabv3_resnet101: https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth


## Citation
If you use our NEL-DM network or BDD-AD dataset in an academic work, please cite:
```
@ARTICLE{10144484,
  author={Feng, Yuchao and Hua, Wei and Sun, Yuxiang},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={NLE-DM: Natural-Language Explanations for Decision Making of Autonomous Driving Based on Semantic Scene Understanding}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2023.3273547}}
```


If you have any questions, pleas feel free to contact us!

Contact: yx.sun@polyu.edu.hk

Website: https://yuxiangsun.github.io/
