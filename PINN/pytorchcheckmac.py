import torch
import torch.nn as nn
import torch.optim as optim

print(f"PyTorch version: {torch.__version__}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else device
print(f"Using device: {device}")