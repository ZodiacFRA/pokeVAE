import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUFA_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == "cuda" else {}

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 40
LOG_INTERVAL = 10

LATENT_SPACE_SIZE = 2

WARMUP_TIME = 10
