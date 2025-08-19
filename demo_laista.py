# =============================================================================
# Contenido para: demo_laista.py
# =============================================================================
r"""
Demo de Inferencia con LAISTA
===================================================
Este script carga un modelo LAISTA pre-entrenado desde un archivo
de checkpoint y lo utiliza para reconstruir una imagen de muestra
del dataset FashionMNIST.

El script:
1. Carga la configuración del experimento desde el archivo .yaml.
2. Carga una imagen de muestra del dataset de prueba.
3. Reconstruye el modelo de adquisición (SPC) y el modelo LAISTA.
4. Carga los pesos entrenados desde el archivo de checkpoint.
5. Realiza la reconstrucción (inferencia) de la imagen.
6. Visualiza la imagen original, la medición y la reconstrucción final.
"""

# %%
# 1. Imports y Configuración del Entorno
# -----------------------------------------------
import os
import torch
import yaml
import matplotlib.pyplot as pltp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- Módulos de nuestro proyecto ---
from src.model import Laista
from src.utils import get_hadamard_patterns
from colibri.optics import SPC
from colibri.recovery.terms.transforms import DCT2D
from colibri.recovery.terms.prior import Sparsity, Denoiser
from colibri.recovery.terms.fidelity import L2

# --- Configuración del Dispositivo ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# %%
# 2. Cargar Configuración y Datos
# -----------------------------------------------
# Carga la misma configuración que usaste para entrenar para asegurar consistencia
config_path = 'configs/spc_fashionmnist.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Ruta a tu mejor modelo guardado
# Asegúrate de que este nombre coincida con el guardado en train.py
checkpoint_name = f"best_model_{config['wandb']['name']}.pth"
checkpoint_path = os.path.join('checkpoints', checkpoint_name)

# Cargar una imagen de muestra del dataset de prueba
transform = transforms.Compose([
    transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
    transforms.ToTensor()
])
test_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Tomemos una imagen específica para la demo (ej. la imagen #18)
sample_iterator = iter(test_loader)
for _ in range(17):
    sample, _ = next(sample_iterator)
sample = sample.to(device)

print("Imagen de muestra cargada. Forma:", sample.shape)
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"No se encontró el archivo de checkpoint en '{checkpoint_path}'. Asegúrate de haber entrenado el modelo primero.")


# %%
# 3. Construir Modelos y Cargar Checkpoint
# -----------------------------------------------
# A. Reconstruir el modelo de adquisición (debe ser idéntico al del entrenamiento)
img_h, img_w = config['data']['img_size'], config['data']['img_size']
n_measurements = int(config['acquisition']['n_measurements_ratio'] * img_h * img_w)
initial_ca = get_hadamard_patterns(img_h, img_w, n_measurements)

acquisition_config = {
    'input_shape': tuple(config['acquisition']['input_shape']),
    'n_measurements': n_measurements,
    'initial_ca': initial_ca
}
acquisition_model = SPC(**acquisition_config).to(device)

fidelity = L2()
prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download_lipschitz", 'device': device}).to(device)

# B. Instanciar el modelo LAISTA con la misma arquitectura
model = Laista(
    acquistion_model=acquisition_model,
    fidelity = fidelity,
    prior = prior
    **config['laista_params'],
    **config['net_params']
).to(device)

# C. Cargar los pesos entrenados desde el checkpoint
print(f"Cargando checkpoint desde: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # ¡Muy importante! Poner el modelo en modo de evaluación

print("Modelo LAISTA cargado y listo para inferencia.")


# %%
# 4. Realizar la Reconstrucción (Inferencia)
# -----------------------------------------------
y = acquisition_model(sample)
x0 = acquisition_model.forward(y, type_calculation="backward")
x_hat = model(y, x0=x0, gt=sample, verbose=True)

# %%
# 5. Visualizar los Resultados (CORREGIDO)
# -----------------------------------------------
import math # Asegúrate de que math esté importado

basis = DCT2D()
theta = basis.forward(x_hat).detach()
normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)) if x.max() > x.min() else x

# --- LÓGICA DE VISUALIZACIÓN CORREGIDA ---
# 1. Calcular el lado del nuevo lienzo cuadrado
side_length = int(math.sqrt(n_measurements)) + 1
new_total_size = side_length * side_length

# 2. Crear un nuevo tensor (lienzo) lleno de ceros
padded_y = torch.zeros(new_total_size, device=y.device)

# 3. Copiar las mediciones originales al principio del lienzo
padded_y[:n_measurements] = y.flatten()

# 4. Ahora sí podemos remodelar el lienzo a un cuadrado 2D para visualización
y_display = padded_y.reshape(1, 1, side_length, side_length)
# --- FIN DE LA CORRECCIÓN ---


plt.figure(figsize=(16, 5))
plt.suptitle(f"Demo de Inferencia LAISTA (max_iters={config['laista_params']['max_iters']})", fontsize=16)

# ... (El resto del código de ploteo se mantiene igual) ...
# Imagen Original
plt.subplot(1, 4, 1)
plt.title("Original (GT)")
plt.imshow(sample.cpu().squeeze().numpy(), cmap="gray")
plt.axis('off')

# Representación Esparsa
plt.subplot(1, 4, 2)
plt.title("Representación Esparsa (DCT)")
plt.imshow(abs(normalize(theta).cpu().squeeze().numpy()), cmap="gray")
plt.axis('off')

# Medición
plt.subplot(1, 4, 3)
plt.title(f"Medición ({n_measurements} puntos)")
plt.imshow(normalize(y_display).cpu().squeeze().numpy(), cmap="gray")
plt.axis('off')

# Reconstrucción
plt.subplot(1, 4, 4)
plt.title("Reconstrucción LAISTA")
plt.imshow(x_hat.cpu().squeeze().detach().numpy(), cmap="gray")
plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
