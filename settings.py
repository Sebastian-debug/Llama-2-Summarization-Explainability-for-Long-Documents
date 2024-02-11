import torch

def initialize_device():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        print("CUDA not available")

initialize_device()
