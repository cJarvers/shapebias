from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

def load_resnet50(layers):
    weights = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=weights)
    subnet = create_feature_extractor(net, return_nodes={layer: layer for layer in layers})
    return subnet