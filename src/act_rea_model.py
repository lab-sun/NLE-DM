from collections import OrderedDict
from typing import Dict, List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .resnet_backbone import resnet50

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # Rebuild the backbone and delete all unused modules
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class DeepLabV3(nn.Module):
    """
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        deeplabhead (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training

    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, classifier_neck, classifier_action, classifier_reason, aux_classifier=None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier_neck = classifier_neck
        self.aux_classifier = aux_classifier
        self.classifier_action = classifier_action
        self.classifier_reason = classifier_reason

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # contract: features is a dict of tensors
        result = OrderedDict()
        features = self.backbone(x)
        x = features["out"]
        x = self.classifier(x)
        x = self.classifier_neck(x)
        y1 = self.classifier_action(x)
        y2 = self.classifier_reason(x)

        result["out"] = [y1, y2]

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            result["aux"] = x

        return result["out"]


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36])
        )


class DeeplabNeck(nn.Sequential):
    def __init__(self):
        super(DeeplabNeck, self).__init__(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(32*18*64, 64, bias=True),
        )


class ActionHead(nn.Sequential):
    def __init__(self, in_channels=64, action_classes=4):
        super(ActionHead, self).__init__(
            nn.Linear(in_channels, action_classes, bias=True)
        )


class ReasonHead(nn.Sequential):
    def __init__(self, in_channels=64, reason_classes=21):
        super(ReasonHead, self).__init__(
            nn.Linear(in_channels, reason_classes, bias=True)
        )


def act_exp(aux=False, num_classes=(4, 21), pretrain_backbone=False):

    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # load the pretrained weights of backbone, set as False
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes)
    classifier_neck = DeeplabNeck()
    classifier_action = ActionHead()
    classifier_reason = ReasonHead()

    model = DeepLabV3(backbone, classifier, classifier_neck, classifier_action, classifier_reason, aux_classifier)

    return model

