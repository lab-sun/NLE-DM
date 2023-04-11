# NLE-DM
The official pytorch implementation of NLE-DM.

## Usage
* Clone this repo and prepare the environment.
```
git clone https://github.com/lab-sun/NLE-DM.git
cd NLE_DM
conda env create -f environment.yml --name NLE_DM
conda activate NLE_DM
```

* Download the dataset, release it and put into the file of `Data`;
```
BDD10K for semantic segmentation
BDD_OIA for jointly predict actions and explanations
BDD_AD for jointly predict actions and descriptions
```
* Download the pretrained weight and put into the file of `weight` (optional);

* To train the network, select the appropriate .py in the folder of `train`
```
pre_train.py: To predict the semantic segmentation results.
train_act_exp.py: 
train_act_des.py:
train_act_des_ablation.py:
```
* To produce the prediction results, select the appropriate .py in the folder of `predict`
```
pre_train.py: To train the deeplabv3 on BDD10K dataset.
train_act_rea.py: To train the network to jointly predict actions and reasons.
train_act_des.py: To train the network to jointly predict actions and descriptions.
train_act_des_ablation.py: To train the network with diffrent combination of action and description.
```

## Dataset
Download the datasets and then extract it in the file of `Data`
* For BDD_AD dataset, please refer to: https: //
* For BDD100K dataset, please refer to: https://www.bdd100k.com/
* For BDD_OIA dataset, please refer to: https://github.com/Twizwei/bddoia_project/blob/master/README.md

## Pretrained weights：
* Download the pretrained weights and then extract it in the file of `weight`
* The link for pretrained weights is: https: //
* The introduction for each weights is as follows:
```
deeplab_bdd10k: network to predict semantic segmentation on BDD10K
act_rea.pth: network to jointly predict actions and reasons
act_des_resnet50.pth: network to jointly predict actions and descriptions (backbone: ResNet 50)
act_des_resnet101.pth: network to jointly predict actions and descriptions (backbone: ResNet 101)
act_des_mobilenetS.pth: network to jointly predict actions and descriptions (backbone: MobileNetV3_Small)
act_des_mobilenetL.pth: network to jointly predict actions and descriptions (backbone: MobileNetV3_Large)
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
bib tex
```


If you have any questions, pleas feel free to contact us!

Contact: yx.sun@polyu.edu.hk

Website: https://yuxiangsun.github.io/
