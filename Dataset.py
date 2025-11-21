import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import random
import copy

class DatasetUnion(Dataset):
    def __init__(self, path_dataset, classes, transform=None, target_transform=None,exepts = None):
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        # Mapeia o nome da classe (Normal, Anormal) -> índice (0, 1, 2...)
        self.class_to_idx = {classe: i for i, classe in enumerate(classes.keys())}

        # Monta os caminhos de cada classe
        self.paths = {}
        for classe, pastas in classes.items():
            all_images = []
            for pasta in pastas:
                full_path = os.path.join(path_dataset, pasta)
                if not os.path.exists(full_path):
                    print(f"Aviso: {full_path} não existe")
                    continue
                imgs = [os.path.join(full_path, f) for f in os.listdir(full_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                all_images.extend(imgs)
            random.shuffle(all_images)
            self.paths[classe] = all_images

        # Balanceia as classes pelo mínimo número de imagens
        min_len = min(len(v) for v in self.paths.values())
        for classe, imgs in self.paths.items():
            label = self.class_to_idx[classe]
            for img_path in imgs[:min_len]:
                self.samples.append((img_path, label))

        random.shuffle(self.samples)
        samples = []
        if exepts is not None:
            for img_path, label in  self.samples:
                if img_path not in exepts:
                    samples.append((img_path, label))
            self.samples = samples
        print(f"Total de imagens balanceadas: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    def distribuicao(self,train,val,test=None):
        class sapleData(Dataset):
            def __init__(self,samples,transform = None,target_transform = None):
                self.transform = transform
                self.target_transform = target_transform
                self.samples = samples
            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
              img_path, label = self.samples[idx]
              image = read_image(img_path).float() / 255.0

              # se for grayscale (1 canal), repete 3x pra virar RGB
              if image.shape[0] == 1:
                  image = image.repeat(3, 1, 1)

              if self.transform:
                  image = self.transform(image)
              if self.target_transform:
                  label = self.target_transform(label)

              return image, label
        setTrain = sapleData(self.samples[:train],transform = copy.deepcopy(self.transform),target_transform=copy.deepcopy(self.target_transform))
        setVal = sapleData(self.samples[train:train+val],transform = copy.deepcopy(self.transform),target_transform=copy.deepcopy(self.target_transform))
        setTest = sapleData(self.samples[train+val:],transform = copy.deepcopy(self.transform),target_transform=copy.deepcopy(self.target_transform))
        return setTrain,setVal,setTest




    def plot(self):
        # Conta imagens por classe dentro de self.samples
        counts = {}
        for img_path, label in self.samples:
            for classe, idx in self.class_to_idx.items():
                if idx == label:
                    counts[classe] = counts.get(classe, 0) + 1

        # Exibe a contagem no terminal
        for classe, count in counts.items():
            print(f"{classe}: {count}")

        # --- Plot ---
        plt.figure(figsize=(8, 5))
        plt.bar(counts.keys(), counts.values())

        plt.title('Distribuição de imagens por classe (balanceadas)', fontsize=14)
        plt.xlabel('Classe', fontsize=12)
        plt.ylabel('Número de imagens', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()
