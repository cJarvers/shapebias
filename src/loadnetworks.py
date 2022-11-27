import torch
from torchvision.models import resnet50, ResNet50_Weights, vgg19, VGG19_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor

###########################
# Standard layers to load #
###########################
# The following lists specify which layers are loaded by default if the
# `layers` argument has value ["default"].
resnet50_layers = ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
vgg19_layers = ["features.3", "features.8", "features.18", "features.26", "features.35", "avgpool", "classifier.1", "classifier.4", "classifier.6"]
vit_b_16_layers = [f"encoder.layers.encoder_layer_{i}.add_1" for i in range(12)] + ["heads.head"]


##################################
# Functions for loading networks #
##################################
def loadnetwork(name, layers, pretrained=True):
    if name == "resnet50":
        net, layers = load_resnet50(layers, pretrained)
    elif name == "vgg19":
        net, layers = load_vgg19(layers, pretrained)
    elif name == "vit" or name == "vit_b_16":
        net, layers = load_vit(layers, pretrained)
    elif name == "shape_resnet":
        net, layers = load_shaperesnet(layers, pretrained)
    else:
        raise(ValueError(f"Network {name} not implemented."))
    return net, layers

def load_resnet50(layers, pretrained=True):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
    net = resnet50(weights=weights)
    if layers is not None:
        if layers == ["default"]:
            layers = resnet50_layers
        net = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return net, layers

def load_shaperesnet(layers, pretrained=True):
    net = resnet50(weights=None)
    if pretrained:
        url = "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(url)
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        net.load_state_dict(state_dict)
    if layers is not None:
        if layers == ["default"]:
            layers = resnet50_layers
        net = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return net, layers


def load_vgg19(layers, pretrained=True):
    if pretrained:
        weights = VGG19_Weights.IMAGENET1K_V1
    else:
        weights = None
    net = vgg19(weights=weights)
    if layers is not None:
        if layers == ["default"]:
            layers = vgg19_layers
        net = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return net, layers

def load_vit(layers, pretrained=True):
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = None
    net = vit_b_16(weights=weights)
    if layers is not None:
        if layers == ["default"]:
            layers = vit_b_16_layers
        net = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return net, layers