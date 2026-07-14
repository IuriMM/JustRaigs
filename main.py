import torch
import numpy as np
import warnings
import questionary
import sys

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
            num_devices = torch_directml.device_count()
            if num_devices > 1:
                choices = [f"{i}: {torch_directml.device_name(i)}" for i in range(num_devices)]
                choice = questionary.select(
                    "Múltiplas GPUs detectadas. Selecione qual usar:",
                    choices=choices
                ).ask()
                
                if choice is None:
                    sys.exit(0)
                    
                selected_idx = int(choice.split(":")[0])
                device = torch_directml.device(selected_idx)
                device_name = f"DirectML ({device} - {torch_directml.device_name(selected_idx)})"
            else:
                device = torch_directml.device(0)
                device_name = f"DirectML ({device} - {torch_directml.device_name(0)})"
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_name = str(device)
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = str(device)
        
    print(f"Usando dispositivo: {device_name}")
    return device

def main():
    choice = questionary.select(
        "Selecione o modo de execução:",
        choices=[
            "All (tudo - padrão)",
            "Train (só treinar)",
            "Eval (só avaliar)",
            "Analyze (visualizar classes)"
        ]
    ).ask()
    
    if choice is None:
        print("Execução cancelada.")
        sys.exit(0)
        
    mode_map = {
        "Train (só treinar)": "train",
        "Eval (só avaliar)": "eval",
        "Analyze (visualizar classes)": "analyze",
        "All (tudo - padrão)": "all"
    }
    
    mode = mode_map[choice]
    device = setup_environment()
    
    # 1. Preparação de Dados
    data_manager = DataManager()
    data_manager.load_data()
    train_loader, val_loader = data_manager.prepare_loaders()

    # 2. Configuração de Modelo
    g_model = GlaucomaModel(num_classes=len(data_manager.classes), device=device)
    model = g_model.get_model()

    # 3. Execução Baseada no Argumento
    if mode in ['analyze', 'all']:
        evaluator = Evaluator(model, val_loader, data_manager.classes, data_manager.idx_to_class, device)
        evaluator.visualize_class_distribution(data_manager.samples)
                
    if mode in ['train', 'all']:
        epochs_input = questionary.text(
            "Quantas épocas deseja treinar?",
            default=str(Config.EPOCHS)
        ).ask()
        
        if epochs_input is None:
            sys.exit(0)
            
        try:
            epochs = int(epochs_input)
        except ValueError:
            epochs = Config.EPOCHS
            
        resume = False
        checkpoint_path = Config.ARTIFACTS_DIR / "checkpoint.pth"
        if checkpoint_path.exists():
            resume_choice = questionary.select(
                "Checkpoint encontrado! Deseja continuar o treino anterior ou começar do zero?",
                choices=["Continuar treino", "Começar do zero"]
            ).ask()
            
            if resume_choice is None:
                sys.exit(0)
            
            resume = (resume_choice == "Continuar treino")
            
        trainer = Trainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            device=device,
            train_samples=data_manager.train_samples,
            num_classes=len(data_manager.classes),
            idx_to_class=data_manager.idx_to_class,
            resume=resume,
            epochs=epochs
        )
        trainer.train()

    if mode in ['eval', 'all']:
        evaluator = Evaluator(model, val_loader, data_manager.classes, data_manager.idx_to_class, device)
        evaluator.evaluate()

if __name__ == '__main__':
    main()
