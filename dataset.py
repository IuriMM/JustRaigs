import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from config import Config

class GlaucomaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class DataManager:
    def __init__(self):
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.train_samples = []
        self.val_samples = []

    def load_data(self):
        print("🔍 [DataManager] Mapeando imagens...")
        image_path_map = {}
        for root, _, files in os.walk(Config.DATA_DIR):
            for f in files:
                if f.endswith(".JPG"):
                    image_path_map[f.replace(".JPG", "")] = Path(root) / f

        print(f"✅ [DataManager] Encontradas {len(image_path_map)} imagens.")

        df = pd.read_csv(Config.LABELS_CSV, sep=";")
        print(f"✅ [DataManager] CSV carregado: {len(df)} registros.")

        self.classes = sorted(df['Final Label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        for _, row in df.iterrows():
            eye_id = row['Eye ID']
            if eye_id in image_path_map:
                self.samples.append((str(image_path_map[eye_id]), self.class_to_idx[row['Final Label']]))

        print(f"✅ [DataManager] Amostras válidas para treinamento/validação: {len(self.samples)}")

    def prepare_loaders(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_samples, self.val_samples = train_test_split(
            self.samples, test_size=0.2, random_state=Config.SEED, stratify=[s[1] for s in self.samples]
        )

        train_dataset = GlaucomaDataset(self.train_samples, transform=train_transform)
        val_dataset = GlaucomaDataset(self.val_samples, transform=val_transform)

        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

        print(f"📊 [DataManager] Treino: {len(train_dataset)} amostras | Validação: {len(val_dataset)} amostras")
        
        return train_loader, val_loader
