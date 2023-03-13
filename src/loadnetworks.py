import torch
from torchvision.models import resnet50, ResNet50_Weights, vgg19, VGG19_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models import alexnet, AlexNet_Weights, googlenet, GoogLeNet_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import bagnets.pytorchnet
import cornet

###########################
# Standard layers to load #
###########################
# The following lists specify which layers are loaded by default if the
# `layers` argument has value ["default"].
alexnet_layers = ["featuers.2", "features.5", "features.8", "features.12", "avgpool", "classifier.1", "classifier.4", "classifier.6"]
cornet_layers = ["V1", "V2", "V4", "IT", "decoder"]
resnet50_layers = ["layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
vgg19_layers = ["features.3", "features.8", "features.18", "features.26", "features.35", "avgpool", "classifier.1", "classifier.4", "classifier.6"]
vgg19_nicenames = ["conv2", "conv4", "conv8", "conv12", "conv16", "avgpool", "fc1", "fc2", "fc3"]
vit_b_16_layers = [f"encoder.layers.encoder_layer_{i}.add_1" for i in range(12)] + ["heads.head"]
vit_nicenames = [f"encoder{i}" for i in range(12)] + ["head"]


##################################
# Functions for loading networks #
##################################
def loadnetwork(name, layers, device="cpu", pretrained=True):
    if name == "resnet50":
        net, layers = load_resnet50(layers, pretrained)
    elif name == "alexnet":
        net, layers = load_alexnet(layers, pretrained)
    elif name == "vgg19":
        net, layers = load_vgg19(layers, pretrained)
    elif name == "vit" or name == "vit_b_16":
        net, layers = load_vit(layers, pretrained)
    elif name == "shape_resnet":
        net, layers = load_shaperesnet(layers, pretrained)
    elif name == "bagnet17":
        net, layers = load_bagnet(layers, pretrained, device=device)
    elif name == "cornet":
        net, layers = load_cornet(layers, pretrained, device=device)
    else:
        raise(ValueError(f"Network {name} not implemented."))
    if not (name == "bagnet17" or name == "cornet"):
        net = net.to(device)
    return net, layers

def load_alexnet(layers, pretrained=True):
    if pretrained:
        weights = AlexNet_Weights.IMAGENET1K_V1
    else:
        weights = None
    net = alexnet(weights=weights)
    if layers is not None:
        if layers == ["default"]:
            layers = alexnet_layers
        net = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return net, layers

def load_bagnet(layers, pretrained=True, device="cpu"):
    net = bagnets.pytorchnet.bagnet17(pretrained=pretrained)
    net = net.to(device)
    if layers is not None:
        # tracing does not work on bagnet. Thus, create_feature_extractor throws an error.
        # This manual workaround extracts activations of standard layers, even if it's a little hacky.
        flat = torch.nn.Flatten()
        def extractor(x):
            x = net.relu(net.bn1(net.conv2(net.conv1(x))))
            results = {}
            results["layer1"] = net.layer1(x)
            results["layer2"] = net.layer2(results["layer1"])
            results["layer3"] = net.layer3(results["layer2"])
            results["layer4"] = net.layer4(results["layer3"])
            results["avgpool"] = torch.nn.AvgPool2d(results["layer4"].size()[2], stride=1)(results["layer4"])
            results["fc"] = net.fc(flat(results["avgpool"]))
            return results
        return extractor, resnet50_layers
    else:
        return net, layers

def load_cornet(layers, pretrained=True, device="cpu"):
    net = cornet.cornet_s(pretrained=True, map_location=device)
    net = net.module.to(device) # unwrap from DataParallel
    if layers is not None:
        # create_feature_extractor does not trace CORnet correctly -> workaround
        def extractor(x):
            results = {}
            results["V1"] = net.V1(x)
            results["V2"] = net.V2(results["V1"])
            results["V4"] = net.V4(results["V2"])
            results["IT"] = net.IT(results["V4"])
            results["decoder"] = net.decoder(results["IT"])
            return results
        return extractor, cornet_layers
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


########################################################
# Helper for nice printing of network names in figures #
########################################################
def netnamenice(name):
    if name == "resnet50":
        return "ResNet-50"
    elif name == "alexnet":
        return "AlexNet"
    elif name == "vgg19":
        return "VGG-19"
    elif name == "vit" or name == "vit_b_16":
        return "ViT"
    elif name == "shape_resnet":
        return "Shape-ResNet"
    elif name == "bagnet17":
        return "BagNet-17"
    elif name == "cornet":
        return "CORnet-S"
    else:
        raise(ValueError(f"Network {name} not implemented."))