import os

class Config:
    API_PORT = 5000
    MODEL_PATH = "models/defect_model.h5"
    CAUSAL_MODEL_PATH = "models/causal_model.pkl"
    DATA_PATH = "data/sample_defects"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
