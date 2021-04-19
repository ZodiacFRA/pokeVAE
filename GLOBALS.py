import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUFA_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == "cuda" else {}

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000
LOG_INTERVAL = 50

LATENT_SPACE_SIZE = 8

WARMUP_TIME = 100
