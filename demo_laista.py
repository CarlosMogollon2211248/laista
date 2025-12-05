import os
import random
import yaml

import numpy as np
import torch
import torch.nn as nn
import wandb
import math

# --- Modulos del proyecto ---
from src.dataset import get_dataloaders
from src.model import Laista
from src.train_test import evaluate, train_one_epoch
from src.utils import get_hadamard_patterns  # Importamos la función para los patrones

# --- Modulos de colibri ---
from colibri.optics import SPC
from colibri.recovery.terms.prior import Sparsity, Denoiser
from colibri.recovery.terms.fidelity import L2
import matplotlib.pyplot as plt
from colibri.recovery.terms.transforms import DCT2D

def set_seed(seed):
    """Fija las semillas de aleatoriedad para que los experimentos sean reproducibles."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Estas dos últimas líneas aseguran un comportamiento determinista en CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config_path='configs/spc_fashionmnist.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

set_seed(config['seed'])
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=config['data']['batch_size'],
    img_size=config['data']['img_size']
    )

img_h, img_w = config['data']['img_size'], config['data']['img_size']
n_measurements = int(config['acquisition']['n_measurements_ratio'] * img_h * img_w)

# Función de ayuda para generar los patrones
initial_ca = get_hadamard_patterns(img_h, img_w, n_measurements)

acquisition_config = {
    'input_shape': tuple(config['acquisition']['input_shape']),
    'n_measurements': n_measurements,
    'initial_ca': initial_ca  # Pasamos los patrones pre-calculados
    }

acquisition_model = SPC(**acquisition_config).to(device)

fidelity = L2()
prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download_lipschitz", 'device': device}).to(device)

model = Laista(
    acquistion_model=acquisition_model,
    fidelity = fidelity,
    prior = prior,
    **config['laista_params'],
    **config['net_params'],
    device= device
    ).to(device)

checkpoints_dir = 'checkpoints'
os.makedirs(checkpoints_dir, exist_ok=True)
best_model_path = os.path.join(checkpoints_dir, f"best_model_{config['wandb']['name']}.pth")

checkpoint = torch.load(best_model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
# ¡CORRECCIÓN AQUÍ! Desempaquetar la tupla
sample = next(iter(test_loader))[0].to(device)#[:1]#.unsqueeze(0)
print(sample.shape)
y = acquisition_model(sample,type_calculation='forward')
x0 = acquisition_model(y,type_calculation='backward')
x_hat, _ = model(y, x0=x0, gt=sample, verbose=True)

acquisition_name = 'spc'

ratio = 0.7
if acquisition_name == "spc":
    n_measurements = int(ratio*(32**2))
    print('numero mediciones', n_measurements)
    n_measurements_sqrt = int(math.sqrt(n_measurements))
    target_size = n_measurements_sqrt ** 2 
    acquisition_config["n_measurements"] = n_measurements

basis = DCT2D()

theta = basis.forward(x_hat).detach()

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

plt.figure(figsize=(10, 10))

plt.subplot(1, 4, 1)
plt.title("Reference")
plt.imshow(sample[0, :, :].cpu().permute(1, 2, 0), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 2)
plt.title("Sparse Representation")
plt.imshow(abs(normalize(theta[0, :, :])).cpu().permute(1, 2, 0), cmap="gray")
plt.xticks([])
plt.yticks([])


if acquisition_name == "spc":
    y = y[:, :target_size, :]  # Recortar para hacer el re shape
    y = y.reshape(y.shape[0], 1, n_measurements_sqrt, n_measurements_sqrt) # Mejor forma para visualizar

plt.subplot(1, 4, 3)
plt.title("Measurement")
plt.imshow(normalize(y[0, :, :]).cpu().permute(1, 2, 0).detach().numpy(), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 4)
plt.title("Reconstruction")
plt.imshow(normalize(x_hat[0, :, :]).cpu().permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.show()