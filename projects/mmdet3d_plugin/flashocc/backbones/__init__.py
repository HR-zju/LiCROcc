from mmdet.models.backbones import ResNet
from .resnet import CustomResNet
from .radar_backbone import PtsBackbone

# from .swin import SwinTransformer

__all__ = ['ResNet', 'CustomResNet', ]
