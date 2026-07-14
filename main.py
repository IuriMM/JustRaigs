import argparse
import torch
import numpy as np
import warnings

# Módulos customizados
from config import Config
from dataset import DataManager
from model import GlaucomaModel
from trainer import Trainer
from evaluator import Evaluator

warnings.filterwarnings('ignore', category=FutureWarning)

def setup_environment():
    Config.setup()
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    device_name = "cpu"
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            device_name = f"DirectML ({device})"
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_name = str(device)
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = str(device)
        
    print(f"🚀 Usando dispositivo: {device_name}")
    return device

def main():
    parser = argparse.ArgumentParser(description="JustRaigs Glaucoma Classification Modules")
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'eval', 'analyze', 'all'],
                        help="Modo de execução: 'train' (só treinar), 'eval' (só avaliar), 'analyze' (visualizar classes) ou 'all' (tudo).")
    args = parser.parse_args()

    print(f"\n====================================")
    print(f" Executando JustRaigs em modo: {args.mode.upper()}")
    print(f"====================================\n")

    device = setup_environment()
    
    # 1. Preparação de Dados
    data_manager = DataManager()
    data_manager.load_data()
    train_loader, val_loader = data_manager.prepare_loaders()

    # 2. Configuração de Modelo
    g_model = GlaucomaModel(num_classes=len(data_manager.classes), device=device)
    model = g_model.get_model()

    # 3. Execução Baseada no Argumento
    if args.mode in ['analyze', 'all']:
        evaluator = Evaluator(model, val_loader, data_manager.classes, data_manager.idx_to_class, device)
        evaluator.visualize_class_distribution(data_manager.samples)
        
    if args.mode in ['train', 'all']:
        trainer = Trainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device,
            train_samples=data_manager.train_samples,
            num_classes=len(data_manager.classes),
            idx_to_class=data_manager.idx_to_class
        )
        trainer.train()

    if args.mode in ['eval', 'all']:
        evaluator = Evaluator(model, val_loader, data_manager.classes, data_manager.idx_to_class, device)
        evaluator.evaluate()

if __name__ == '__main__':
    main()
