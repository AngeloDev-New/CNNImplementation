import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        train_set,
        val_set,
        test_set,
        classes,
        models_path,
        batch_size=32,
        lr=3e-4,
        momentum=0.9,
        weight_decay=1e-4,
    ):
        os.makedirs(models_path, exist_ok=True)
        self.treino_concluido = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.classes = classes
        self.models_path = models_path

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
        self.test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        self.last_epoch = 0
        self.best_val_loss = float("inf")


        self._load_state()


    def _load_state(self):
        json_path = os.path.join(self.models_path, "training_state.json")
        last_path = os.path.join(self.models_path, "last.pth")

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                state = json.load(f)

            self.history = state["history"]
            self.last_epoch = state["last_epoch"]
            self.best_val_loss = state["best_val_loss"]
            self.treino_concluido = state["treino_concluido"]
            if self.treino_concluido:
                print("Treinamento Finalizado")
            else:
                print(f" Estado carregado! Retomando a partir da epoch {self.last_epoch + 1}")
            print(f"   Histórico: {len(self.history['train_loss'])} epochs anteriores")

        if os.path.exists(last_path):
            self.model.load_state_dict(torch.load(last_path, map_location=self.device))
            print("✔ last.pth carregado")



    def _save_state(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.models_path, "last.pth"))

        data = {
            "last_epoch": epoch + 1,  
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }
        data['treino_concluido'] = self.treino_concluido
        with open(os.path.join(self.models_path, "training_state.json"), "w") as f:
            json.dump(data, f, indent=4)



    def train(self, num_epochs, patience=10):
        no_improve = 0
        if self.treino_concluido:
            # print('O treino ja voi concluido')
            return
        for epoch in range(self.last_epoch, num_epochs):
            print("\n-----------------------------------")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("Início:", datetime.now().strftime("%H:%M:%S"))

            self.model.train()
            running_loss = 0
            correct = 0
            total = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            self.model.eval()
            val_loss_sum = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss_sum += loss.item() * inputs.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

            print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
            print(f"val_loss:   {val_loss:.4f}, val_acc:   {val_acc:.4f}")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self._save_state(epoch)

            if val_loss < self.best_val_loss:
                print(f"✔ Novo melhor modelo! (val_loss: {val_loss:.4f})")
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.models_path, "best.pth"))
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"\nEarly stopping ativado! ({patience} epochs sem melhoria)")
                break

        print("\nTreinamento finalizado!")
        self.treino_concluido = True

    def evaluate(self, use_best=True):

        if use_best:
            best_path = os.path.join(self.models_path, "best.pth")
            if os.path.exists(best_path):
                self.model.load_state_dict(torch.load(best_path, map_location=self.device))
                print("Avaliando com best.pth")
            else:
                print("best.pth não encontrado, usando modelo atual")

        self.model.eval()
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                preds = outputs.argmax(1)

                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.numpy())

        acc = np.mean(np.array(preds_all) == np.array(labels_all))
        print(f"\n Test accuracy: {acc:.4f}")

        cm = confusion_matrix(labels_all, preds_all)
        disp = ConfusionMatrixDisplay(cm, display_labels=list(self.classes))
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()


    def plot(self):

        if len(self.history["train_loss"]) == 0:
            print(" Nenhum histórico para plotar")
            return

        history = self.history
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


        ax1.plot(epochs, history["train_loss"], 'b-', label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"], 'r-', label="Val Loss", linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        ax2.plot(epochs, history["train_acc"], 'b-', label="Train Acc", linewidth=2)
        ax2.plot(epochs, history["val_acc"], 'r-', label="Val Acc", linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()