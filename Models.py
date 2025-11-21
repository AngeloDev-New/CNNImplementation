import torchvision.models as models
import torch.nn as nn

def load_model(name, num_classes=2, pretrained=True):
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenetv2":
        model = models.mobilenet_v2(weights="DEFAULT" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "densenet121":
        model = models.densenet121(weights="DEFAULT" if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Modelo '{name}' não suportado.")

    return model
