import torchvision.models as models
import torch.nn as nn
import inspect


def load_model(name, num_classes=2, pretrained=True):
    name = name.lower()

    # 1. Lista todos os modelos disponíveis no torchvision
    available = {k.lower(): v for k, v in models.__dict__.items() 
                 if inspect.isfunction(v) or inspect.isclass(v)}

    if name not in available:
        raise ValueError(
            f"Modelo '{name}' não encontrado!\n"
            f"Modelos disponíveis: {list(available.keys())}"
        )

    constructor = available[name]

    # 2. Carrega pesos
    kwargs = {}
    if pretrained:
        kwargs["weights"] = "DEFAULT"
    else:
        kwargs["weights"] = None

    # 3. Constrói o modelo
    model = constructor(**kwargs)

    # 4. Acha a última camada automaticamente e substitui
    # ResNet, EfficientNet, RegNet, ConvNeXt, ViT, etc
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # MobileNet, EfficientNet, etc: classifier é uma lista/Sequential
    elif hasattr(model, "classifier"):

        # Acha a última Linear dentro de classifier
        for i in reversed(range(len(model.classifier))):
            if isinstance(model.classifier[i], nn.Linear):
                in_f = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_f, num_classes)
                break

    # DenseNet
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_classes)

    else:
        raise ValueError("Não sei substituir a última camada desse modelo.")

    return model
