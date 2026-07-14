import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix, classification_report
)
from config import Config

class Evaluator:
    def __init__(self, model, loader, classes, idx_to_class, device):
        self.model = model
        self.loader = loader
        self.classes = classes
        self.idx_to_class = idx_to_class
        self.device = device

    def visualize_class_distribution(self, samples):
        print("📊 [Evaluator] Gerando visualização da distribuição de classes...")
        counts = pd.Series([s[1] for s in samples]).value_counts().sort_index()
        class_names = [self.idx_to_class[i] for i in counts.index]
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=class_names, y=counts.values, palette="viridis")
        
        for i, count in enumerate(counts.values):
            plt.text(i, count + 5, f"{count}\n({100*count/len(samples):.1f}%)", ha='center', fontweight='bold')
            
        plt.title("Distribuição de Classes no Dataset")
        plt.ylabel("Número de Amostras")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(Config.ARTIFACTS_DIR / "class_distribution.png", dpi=150)
        plt.close()
        
        imbalance_ratio = counts.max() / counts.min()
        print(f"⚖️ [Evaluator] Razão de desbalanceamento: {imbalance_ratio:.2f}x")

    def evaluate(self):
        print("🔍 [Evaluator] Carregando o melhor modelo salvo...")
        try:
            self.model.load_state_dict(torch.load(Config.ARTIFACTS_DIR / "model_best.pth", weights_only=True))
        except FileNotFoundError:
            print("⚠️ [Evaluator] model_best.pth não encontrado. Usando os pesos atuais do modelo.")
            
        self.model.eval()
        
        all_preds, all_labels = [], []
        
        print("▶️ [Evaluator] Iniciando avaliação do conjunto de validação...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.loader):
                outputs = self.model(images.to(self.device))
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                if (i + 1) % 10 == 0 or (i + 1) == len(self.loader):
                    print(f"   [Avaliação] Processado batch {i+1}/{len(self.loader)}")
                
        print("\n" + "="*40)
        print("📊 RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*40)
        print(classification_report(all_labels, all_preds, target_names=self.classes))
        
        print(f"Balanced Accuracy: {balanced_accuracy_score(all_labels, all_preds):.4f}")
        print(f"F1 Macro Score:    {f1_score(all_labels, all_preds, average='macro'):.4f}")
        
        self._plot_confusion_matrix(all_labels, all_preds)

    def _plot_confusion_matrix(self, all_labels, all_preds):
        print("📉 [Evaluator] Gerando Matriz de Confusão...")
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.savefig(Config.ARTIFACTS_DIR / "confusion_matrix.png")
        plt.close()
