import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from config import Config

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, train_samples, num_classes, idx_to_class, resume=False, epochs=Config.EPOCHS):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.resume = resume
        self.epochs = epochs
        
        print("[Trainer] Configurando Loss Function e Optimizer...")
        train_labels = torch.tensor([s[1] for s in train_samples])
        class_counts = torch.bincount(train_labels, minlength=num_classes).float()
        weights = class_counts.sum() / (num_classes * class_counts)
        weights = weights.to(self.device)
        
        self.criterion = FocalLoss(alpha=weights, gamma=2.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        print("Pesos aplicados à Loss Function:")
        for i, w in enumerate(weights.cpu().numpy()):
            print(f"   - {idx_to_class[i]}: {w:.4f}")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        total_batches = len(self.train_loader)
        
        print(f"\nIniciando Época {epoch}/{self.epochs} - Fase de Treino")
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                print(f"   [Treino] Época {epoch} | Batch {i+1}/{total_batches} | Loss parcial: {loss.item():.4f}")
                
        return running_loss / total, correct / total

    def validate(self, epoch):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        total_batches = len(self.val_loader)
        
        print(f"Iniciando Época {epoch}/{self.epochs} - Fase de Validação")
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    print(f"   [Validação] Época {epoch} | Batch {i+1}/{total_batches} | Loss parcial: {loss.item():.4f}")
                    
        return running_loss / total, correct / total

    def train(self):
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 1
        import os
        checkpoint_path = Config.ARTIFACTS_DIR / "checkpoint.pth"

        if self.resume and checkpoint_path.exists():
            print("[Trainer] Carregando checkpoint anterior...")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint['history']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"[Trainer] Retomando da época {start_epoch}!")

        print("[Trainer] Iniciando Treinamento Geral...")
        
        total_epochs = start_epoch + self.epochs - 1
        
        try:
            for epoch in range(start_epoch, total_epochs + 1):
                t0 = time.time()
                
                # Ajusta temporariamente self.epochs para o print do train_one_epoch e validate
                original_epochs = self.epochs
                self.epochs = total_epochs
                
                train_loss, train_acc = self.train_one_epoch(epoch)
                val_loss, val_acc = self.validate(epoch)
                
                self.epochs = original_epochs
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                duration = time.time() - t0
                print(f"Resumo Época {epoch}/{total_epochs} - {duration:.1f}s | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), Config.ARTIFACTS_DIR / "model_best.pth")
                    patience_counter = 0
                    print("[Trainer] Melhor modelo salvo!")
                else:
                    patience_counter += 1
                    if patience_counter >= Config.PATIENCE:
                        print(f"[Trainer] Early Stopping acionado na época {epoch}")
                        break
                
                self.scheduler.step(val_loss)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': history,
                    'best_val_loss': best_val_loss
                }, checkpoint_path)

        except KeyboardInterrupt:
            print("\n[Trainer] Treinamento pausado pelo usuário (Ctrl+C).")
            print("[Trainer] Checkpoint salvo. Você poderá continuar de onde parou depois.")
            return history

        self._plot_history(history)
        print("[Trainer] Treinamento Finalizado!")
        return history

    def _plot_history(self, history):
        print("[Trainer] Gerando gráfico de histórico de treinamento...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss History')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Val')
        plt.title('Accuracy History')
        plt.legend()

        plt.savefig(Config.ARTIFACTS_DIR / "training_history.png")
        plt.close()
