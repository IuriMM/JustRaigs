import torch.nn as nn
from torchvision import models

class GlaucomaModel:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.model = self._build_model()

    def _build_model(self):
        print("⚙️ [GlaucomaModel] Inicializando modelo EfficientNet_V2_S...")
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

        for param in model.features.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, self.num_classes)
        )

        model = model.to(self.device)
        print("✅ [GlaucomaModel] Modelo criado e movido para o dispositivo.")
        return model
        
    def get_model(self):
        return self.model
