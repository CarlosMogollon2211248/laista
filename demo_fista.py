r"""
Demo FISTA/ISTA. Sirve tanto para el ISTA como para el FISTA, lo unico a cambiar seria el algoritmo
===================================================

"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os

from torch.utils import data

print("Current Working Directory ", os.getcwd())

import sys

sys.path.append(os.path.join(os.getcwd()))

# General imports
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(0)

manual_device = "cpu"
# Check GPU support
print("GPU support: ", torch.cuda.is_available())

if manual_device:
    device = manual_device
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Load dataset
# -----------------------------------------------
from colibri.data.datasets import CustomDataset

name = "fashion_mnist"
path = "."
batch_size = 32


dataset = CustomDataset(name, path)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

acquisition_name = 'spc'  # ['spc', 'cassi']

# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

datos = next(iter(dataset_loader))
sample = datos["input"][17]
# print(sample.shape)
sample = sample.unsqueeze(0).to(device)
# print(sample.shape)
# %%
# Optics forward model

import math
from colibri.optics import SPC, SD_CASSI, DD_CASSI, C_CASSI

img_size = sample.shape[1:]
_, M, N = img_size
print(img_size)
acquisition_config = dict(
    input_shape=img_size,
)

ratio = 1 
if acquisition_name == "spc":
    n_measurements = int(ratio*(28**2))
    print('numero mediciones', n_measurements)
    n_measurements_sqrt = int(math.sqrt(n_measurements))
    target_size = n_measurements_sqrt ** 2 
    acquisition_config["n_measurements"] = n_measurements


# CODIGO DE APERTURA (HADAMARD)

import numpy as np
from libs.ordering import get_index_matrix
from libs.ordering.sequency import sequency_order
from libs.row_wise import hadamard_row

n = 10
ordering = "cake_cutting"

size = np.sqrt(2**n).astype(int)
index_matrix = size*size - get_index_matrix(size, ordering)
ordering_list = sequency_order(2**n)

order_temp = index_matrix.copy()
order_temp[:, 1::2] = order_temp[::-1, 1::2]
order_temp = index_matrix.reshape(-1, order="F")
order_temp = np.argsort(order_temp)
ordering_list = [ordering_list[i] for i in order_temp]

H = []


for i in range(2**n):
    index = ordering_list[i]
    H.append(hadamard_row(index, n))


H = np.array(H).squeeze()
initial_hadamard_ca = H[:n_measurements, :M*N]

print("Ordering mode: ", ordering)
print('CA Generado')
acquisition_config['initial_ca'] = initial_hadamard_ca
# acquisition_config['initial_ca'] = None

acquisition_model = {"spc": SPC, "sd_cassi": SD_CASSI, "dd_cassi": DD_CASSI, "c_cassi": C_CASSI}[
    acquisition_name
]

acquisition_model = acquisition_model(**acquisition_config)

y = acquisition_model(sample)

# Reconstruct image
from colibri.recovery.fista import Fista
from colibri.recovery.ista import Ista 
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.transforms import DCT2D

algo_params = {
    "max_iters": 60,
    "alpha": 0.0004774212882981862,
    "_lambda": 0.010969419598768213,
}

fidelity = L2()
prior = Sparsity(basis="dct")

# SELECCIONAR ALGORITMO
# fista = Fista(acquisition_model, fidelity, prior, **algo_params)
ista = Ista(acquisition_model, fidelity, prior, **algo_params)

x0 = acquisition_model.forward(y, type_calculation="backward")
# x_hat = fista(y, gt=sample, x0=x0, verbose=True)
x_hat = ista(y, gt=sample, x0=x0, verbose=True)

basis = DCT2D()

theta = basis.forward(x_hat).detach()

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

plt.figure(figsize=(10, 10))

plt.subplot(1, 4, 1)
plt.title("Reference")
plt.imshow(sample[0, :, :].permute(1, 2, 0), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 2)
plt.title("Sparse Representation")
plt.imshow(abs(normalize(theta[0, :, :])).permute(1, 2, 0), cmap="gray")
plt.xticks([])
plt.yticks([])

if acquisition_name == "spc":
    y = y[:, :target_size, :]  # Recortar para hacer el re shape
    y = y.reshape(y.shape[0], 1, n_measurements_sqrt, n_measurements_sqrt) # Mejor forma para visualizar

plt.subplot(1, 4, 3)
plt.title("Measurement")
plt.imshow(normalize(y[0, :, :]).permute(1, 2, 0).detach().numpy(), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 4)
plt.title("Reconstruction")
plt.imshow(normalize(x_hat[0, :, :]).permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.show()

# %%
