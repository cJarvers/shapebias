from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

def loadnetwork(name, layers, pretrained=True):
    if name == "resnet50":
        net = load_resnet50(layers, pretrained)
    else:
        raise(ValueError(f"Network {name} not implemented."))
    return net

def load_resnet50(layers, pretrained=True):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None
    net = resnet50(weights=weights)
    subnet = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return subnet