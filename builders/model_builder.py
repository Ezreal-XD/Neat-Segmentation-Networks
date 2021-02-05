from model.SQNet import SQNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.UNet import UNet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.CGNet import CGNet
from model.EDANet import EDANet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.LEDNet import LEDNet
from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.ContextNet import ContextNet
from model.FastSCNN import FastSCNN
from model.DABNet import DABNet
from model.FSSNet import FSSNet
from model.FPENet import FPENet
from model.FCN import FCN32VGG
import network


def build_model(model_name, num_classes):
    # for deeplabv3
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    if model_name == 'SQNet':
        return SQNet(classes=num_classes)
    elif model_name == 'LinkNet':
        return LinkNet(classes=num_classes)
    elif model_name == 'SegNet':
        return SegNet(classes=num_classes)
    elif model_name == 'UNet':
        return UNet(classes=num_classes)
    elif model_name == 'ENet':
        return ENet(classes=num_classes)
    elif model_name == 'ERFNet':
        return ERFNet(classes=num_classes)
    elif model_name == 'CGNet':
        return CGNet(classes=num_classes)
    elif model_name == 'EDANet':
        return EDANet(classes=num_classes)
    elif model_name == 'ESNet':
        return ESNet(classes=num_classes)
    elif model_name == 'ESPNet':
        return ESPNet(classes=num_classes)
    elif model_name == 'LEDNet':
        return LEDNet(classes=num_classes)
    elif model_name == 'ESPNet_v2':
        return EESPNet_Seg(classes=num_classes)
    elif model_name == 'ContextNet':
        return ContextNet(classes=num_classes)
    elif model_name == 'FastSCNN':
        return FastSCNN(classes=num_classes)
    elif model_name == 'DABNet':
        return DABNet(classes=num_classes)
    elif model_name == 'FSSNet':
        return FSSNet(classes=num_classes)
    elif model_name == 'FPENet':
        return FPENet(classes=num_classes)
    elif model_name == 'FCN':
        return FCN32VGG(classes=num_classes)
    elif model_name in model_map.keys():
        return model_map[model_name](num_classes, output_stride=8)
