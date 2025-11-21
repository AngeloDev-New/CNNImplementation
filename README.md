# CNN Implementation

Este projeto implementa redes neurais convolucionais (CNNs) para classificação de imagens médicas utilizando PyTorch e Google Colab.

## Pré-requisitos

Antes de começar, certifique-se de ter as seguintes pastas configuradas no Google Drive:
- `imagens_separadas` - Dataset com as imagens organizadas
- `modelos` - Diretório para salvar os modelos treinados

## Instalação e Configuração

### 1. Conectar ao Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clonar o Repositório e Configurar Ambiente

```python
# Remover instalação anterior (se existir)
!rm -rf CNNImplementation/

# Clonar repositório
!git clone https://github.com/AngeloDev-New/CNNImplementation.git

# Importar módulos necessários
from CNNImplementation.Models import load_model
from CNNImplementation.Dataset import DatasetUnion
from CNNImplementation.Transforms import CropSides
from CNNImplementation.trainer import Trainer
import torchvision.transforms as T
import torch
```

## Preparação do Dataset

### Definir Classes

```python
classes = {
    'Normal': ['BI-RADS_1'],
    'Anormal': [
        'BI-RADS_0',
        'BI-RADS_2',
        'BI-RADS_3',
        'BI-RADS_4',
        'BI-RADS_5'
    ]
}
```

### Configurar Transformações

```python
transform = T.Compose([
    CropSides(top=0.20, right=0, bottom=0, left=0),
    T.Resize((224, 224)),
    T.ConvertImageDtype(torch.float),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Carregar e Distribuir Dataset

```python
# Configurar caminho do dataset
path_dataset = '/content/drive/MyDrive/imagens_separadas'
exepts = []

# Criar dataset
dataset = DatasetUnion(
    path_dataset,
    classes,
    transform=transform,
    exepts=exepts
)

# Definir distribuição
total = len(dataset)
n_train = 1600
n_val = 400
n_test = total - n_train - n_val

# Dividir dataset
train_set, val_set, test_set = dataset.distribuicao(
    train=n_train,
    val=n_val
)

# Visualizar distribuição
dataset.plot()
```

## Modelos Disponíveis

O sistema suporta os seguintes modelos:

### AlexNet
- `alexnet`

### ConvNeXt
- `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

### DenseNet
- `densenet121`, `densenet161`, `densenet169`, `densenet201`

### EfficientNet
- `efficientnet_b0` até `efficientnet_b7`
- `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`

### GoogleNet
- `googlenet`

### Inception
- `inception_v3`

### MNASNet
- `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`

### MobileNet
- `mobilenet_v2`
- `mobilenet_v3_large`, `mobilenet_v3_small`

### RegNet
- `regnet_y_400mf`, `regnet_y_800mf`

### ResNet
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

## Treinamento

```python
# Criar trainer
trainer = Trainer(
    model=load_model('resnet50'),
    train_set=train_set,
    val_set=val_set,
    test_set=test_set,
    classes=classes,
    models_path="/content/drive/MyDrive/modelosCeonc/resnet50"
)

# Treinar modelo
trainer.train(num_epochs=50, patience=5)

# Avaliar modelo
trainer.evaluate()

# Visualizar resultados
trainer.plot()
```

## Parâmetros de Treinamento

- **num_epochs**: Número máximo de épocas de treinamento
- **patience**: Número de épocas sem melhoria antes de parar (early stopping)

## Estrutura de Diretórios

```
Google Drive/
├── imagens_separadas/          # Dataset de imagens
│   ├── BI-RADS_0/
│   ├── BI-RADS_1/
│   ├── BI-RADS_2/
│   ├── BI-RADS_3/
│   ├── BI-RADS_4/
│   └── BI-RADS_5/
└── modelosCeonc/               # Modelos treinados
    └── resnet50/
```

## Personalização

Para treinar com um modelo diferente, basta alterar o parâmetro na função `load_model()`:

```python
trainer = Trainer(
    model=load_model('efficientnet_b0'),  # Modelo desejado
    # ... outros parâmetros
)
```

## Notas

- As imagens são redimensionadas para 224x224 pixels
- A normalização usa os valores padrão do ImageNet
- O sistema implementa early stopping para evitar overfitting
- Os modelos são salvos automaticamente durante o treinamento

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests no repositório.

## Licença

Este projeto está sob a licença especificada no repositório.

---

**Desenvolvido por**: Angelo Dev
**Repositório**: [CNNImplementation](https://github.com/AngeloDev-New/CNNImplementation)
