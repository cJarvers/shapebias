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
        net = load_resnet50(layers, pretrained)
    elif name == "vgg19":
        net = load_vgg19(layers, pretrained)
    elif name == "vit" or name == "vit_b_16":
        net = load_vit(layers, pretrained)
    else:
        raise(ValueError(f"Network {name} not implemented."))
    return net

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
    return net

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
    return net

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
    return net