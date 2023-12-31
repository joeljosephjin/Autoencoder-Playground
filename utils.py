import numpy as np
import random
import torch
import os
import joblib


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# set_seed(42)
    
def save_model(model, filename="simple_vae_model.mdl"):
    if not filename.endswith(".mdl"):
        filename = filename+".mdl"
    joblib.dump(model, filename="saved/"+filename)

def load_model(filename="simple_vae_model.mdl"):
    if not filename.endswith(".mdl"):
        filename = filename+".mdl"
    model = joblib.load(filename="saved/"+filename)
    return model

