import os
from pathlib import Path
from PIL import Image
import concurrent.futures
from tqdm import tqdm

# Configurações
SRC_DIR = Path("data")
DEST_DIR = Path("data_resized")
# 512x512 é um excelente tamanho intermediário. 
# Grande o suficiente para o RandomResizedCrop de 224x224 funcionar bem sem perda de qualidade,
# mas pequeno o suficiente para não sobrecarregar a CPU.
TARGET_SIZE = (512, 512) 
QUALITY = 85

def process_image(img_path):
    try:
        # Calcula o caminho relativo para manter a mesma estrutura de pastas (ex: data/0/img.jpg -> data_resized/0/img.jpg)
        rel_path = img_path.relative_to(SRC_DIR)
        dest_path = DEST_DIR / rel_path
        
        # Pula a imagem se ela já existir no destino (útil se o script for interrompido e você rodar de novo)
        if dest_path.exists():
            return True
        
        # Cria as pastas pai caso não existam
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Abre, redimensiona e salva
        with Image.open(img_path) as img:
            img = img.convert("RGB") # Garante que está no formato correto
            # LANCZOS é o melhor algoritmo de interpolação para redução de tamanho
            img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img_resized.save(dest_path, "JPEG", quality=QUALITY)
            
        return True
    except Exception as e:
        print(f"\nErro ao processar {img_path}: {e}")
        return False

def main():
    print(f"🔍 Escaneando a pasta '{SRC_DIR}' em busca de imagens...")
    all_images = list(SRC_DIR.rglob("*.JPG")) + list(SRC_DIR.rglob("*.jpg"))
    print(f"✅ Encontradas {len(all_images)} imagens.")
    
    if not all_images:
        print("Nenhuma imagem encontrada. Verifique se a pasta 'data' está no local correto.")
        return

    # Garante que a pasta base de destino existe
    DEST_DIR.mkdir(exist_ok=True)
    
    print(f"🚀 Redimensionando imagens para {TARGET_SIZE} e salvando em '{DEST_DIR}'...")
    
    # Usa múltiplas threads baseadas nos núcleos do seu processador para fazer isso MUITO rápido
    workers = min(os.cpu_count() or 4, 16)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Mapeia a função para todas as imagens e mostra uma barra de progresso (tqdm)
        list(tqdm(executor.map(process_image, all_images), total=len(all_images), desc="Progresso"))

if __name__ == "__main__":
    main()
