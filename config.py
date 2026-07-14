from pathlib import Path

class Config:
    DATA_DIR = Path("data_resized")
    LABELS_CSV = Path("data_resized/JustRAIGS_Train_labels.csv")
    ARTIFACTS_DIR = Path("artifacts")
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    PATIENCE = 3
    SEED = 42

    @classmethod
    def setup(cls):
        cls.ARTIFACTS_DIR.mkdir(exist_ok=True)
