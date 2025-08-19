r"""
Demo FISTA/ISTA con Optimización de Hiperparámetros (v2)
======================================================
Este script demuestra la reconstrucción de imágenes para un sistema de Single-Pixel Camera (SPC)
y realiza una búsqueda aleatoria para encontrar los hiperparámetros óptimos para el algoritmo de
reconstrucción ISTA, utilizando la función de métrica PSNR de la librería colibri.

El script:
1. Carga el dataset Fashion MNIST.
2. Configura un modelo óptico de SPC con una apertura de codificación basada en patrones de Hadamard.
3. Define un rango de búsqueda para los hiperparámetros del algoritmo ISTA.
4. Ejecuta un bucle para probar combinaciones aleatorias de estos parámetros.
5. Para cada combinación, reconstruye un subconjunto de imágenes y calcula el PSNR promedio
   usando `colibri.metrics.psnr`.
6. Almacena y clasifica los resultados para encontrar la mejor combinación de parámetros.
7. Guarda los 10 mejores resultados en un archivo de texto.
8. Muestra una reconstrucción visual utilizando los mejores parámetros encontrados.
"""

# %%
# 1. Selección de Directorio de Trabajo y Dispositivo
# ----------------------------------------------------
import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import Subset
import collections

# --- Configuraciones Iniciales ---
# Asegúrate de que las librerías 'colibri' y 'libs' estén en el path de Python
print("Current Working Directory ", os.getcwd())
# sys.path.append(os.getcwd()) # Descomentar si es necesario

# General imports
from colibri.data.datasets import CustomDataset
from colibri.optics import SPC
from colibri.recovery.ista import Ista
# from colibri.recovery.fista import Fista # Descomentar si quieres probar con FISTA
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.transforms import DCT2D
from colibri.metrics import psnr  # <-- IMPORTACIÓN ACTUALIZADA
from libs.ordering import get_index_matrix
from libs.ordering.sequency import sequency_order
from libs.row_wise import hadamard_row

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

manual_device = "cpu"
# Check GPU support
print("GPU support: ", torch.cuda.is_available())

if manual_device:
    device = manual_device
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# %%
# 2. Carga del Dataset
# -----------------------------------------------
name = "fashion_mnist"
path = "."
batch_size = 32

new_size = (32, 32)
input_transforms = transforms.Resize(new_size)

# 2. Crear el diccionario de transformaciones.
transform_dict = {
    'input': input_transforms
}

dataset = CustomDataset(name, path, builtin_train=True, transform_dict=transform_dict)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

acquisition_name = 'spc'


# %%
# 3. Configuración del Modelo Óptico (Forward Model)
# -------------------------------------------------
sample_data = next(iter(dataset_loader))
# Accedemos al lote de imágenes con la llave 'input'
sample = sample_data['input'][0].unsqueeze(0).to(device)

img_size = sample.shape[2:]
M, N = img_size
print(f"Image size (H, W): ({M}, {N})")

acquisition_config = dict(
    input_shape=(1, M, N),
)

ratio = 0.3
if acquisition_name == "spc":
    n_measurements = int(ratio * (M * N))
    print('Número de mediciones:', n_measurements)
    acquisition_config["n_measurements"] = n_measurements

# --- CÓDIGO DE APERTURA (HADAMARD) ---
n = 10
ordering = "sequency"

size = int(np.sqrt(2**n))
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
print('CA de Hadamard Generado')
acquisition_config['initial_ca'] = initial_hadamard_ca

acquisition_model = SPC(**acquisition_config)
acquisition_model.to(device)

fidelity = L2()
prior = Sparsity(basis="dct")


# %%
# 4. Búsqueda de Hiperparámetros Óptimos
# ---------------------------------------------

# --- Definir rangos para la búsqueda aleatoria ---
param_ranges = {
    "max_iters": [100, 200, 300, 400, 500],
    "alpha": (1e-5, 1e-3),
    "_lambda": (1e-3, 1e-1),
}

# --- Configuraciones de la búsqueda ---
num_combinations = 30
num_images_per_class = 20
output_filename = f'mejores_parametros/top_results_ratio_{ratio}_ista_ca_sequency.txt'

# test_images = [dataset[i]['input'].unsqueeze(0).to(device) for i in range(num_images_for_avg)]
selected_indices = []
class_counts = collections.defaultdict(int)

print(f"Creando un subset balanceado con {num_images_per_class} imágenes por clase...")

for idx in range(len(dataset)):
    label = dataset[idx]['output'].item()
    # Comprobamos si todavía necesitamos más imágenes de esta clase
    if class_counts[label] < num_images_per_class:
        selected_indices.append(idx)
        class_counts[label] += 1
    # Si ya tenemos suficientes imágenes de todas las clases, paramos.
    if len(selected_indices) >= num_images_per_class * 10:
        break

test_images = Subset(dataset, selected_indices)

print(f'Subset de {len(test_images)}')

labels = [dataset[i]['output'].item() for i in selected_indices]
print('Numero de imagenes por clase')
print(collections.Counter(labels))

results = []
print(f"\n--- Iniciando Búsqueda Aleatoria de Hiperparámetros para ISTA ---")
print(f"Probando {num_combinations} combinaciones, cada una evaluada en {num_images_per_class*10} imágenes.")

for j in range(num_combinations):
    algo_params = {
        "max_iters": random.choice(param_ranges['max_iters']),
        "alpha": random.uniform(*param_ranges["alpha"]),
        "_lambda": random.uniform(*param_ranges["_lambda"])
    }
    print(f'\n[Prueba {j+1}/{num_combinations}] Parámetros: {algo_params}')

    psnr_values = []
    is_valid_combination = True
    for i, gt_item in enumerate(test_images):
        gt_sample = gt_item['input'].unsqueeze(0).to(device)
        y = acquisition_model(gt_sample)
        
        # ALGORITMO SELECCIONADO: ISTA
        ista_solver = Ista(acquisition_model, fidelity, prior, **algo_params)
        
        x0 = acquisition_model.forward(y, type_calculation="backward")
        
        x_hat = ista_solver(y, x0=x0, verbose=False)

        # Normalizar la reconstrucción al rango de la imagen original [0, 1]
        # Es importante para que el PSNR sea comparable entre diferentes reconstrucciones.
        if x_hat.max() > x_hat.min():
            x_hat_norm = (x_hat - x_hat.min()) / (x_hat.max() - x_hat.min())
        else:
            x_hat_norm = x_hat # Evitar división por cero si la imagen es plana

        # Calcular PSNR usando la función de colibri
        # La imagen original (gt_sample) ya está en el rango [0, 1]
        # Por lo tanto, el data_range correcto es 1.0.
        psnr_val = psnr(y_true=gt_sample, y_pred=x_hat_norm, data_range=1.0)
        psnr_val = psnr_val.item() # Extraer el valor flotante del tensor

        if np.isnan(psnr_val) or np.isinf(psnr_val):
            print(f"PSNR fue NaN/Inf en la imagen {i+1}. Saltando esta combinación de parámetros.")
            is_valid_combination = False
            break
        
        psnr_values.append(psnr_val)

    if is_valid_combination:
        psnr_mean = np.mean(psnr_values)
        print(f'PSNR promedio: {psnr_mean:.4f} dB')
        results.append((algo_params, psnr_mean))
    else:
        print(f'Combinación {j+1} descartada por producir resultados no válidos.')

# %%
# 5. Reporte y Almacenamiento de Resultados
# ----------------------------------------------
print("\n--- Búsqueda de Parámetros Finalizada ---")

if not results:
    print("No se encontraron combinaciones de parámetros válidas.")
else:
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    top_results = sorted_results[:10]

    print("\n--- Mejores 10 Combinaciones de Parámetros Encontradas ---")
    for res in top_results:
        print(f"Parámetros: {res[0]}, PSNR Promedio: {res[1]:.4f} dB")

    with open(output_filename, 'w') as f:
        f.write("Mejores 10 resultados de la búsqueda de hiperparámetros para ISTA con apertura Hadamard\n")
        f.write("=====================================================================================\n")
        for i, (params, psnr_val) in enumerate(top_results):
            f.write(f"#{i+1}: PSNR Promedio = {psnr_val:.4f} dB\n")
            f.write(f"   Parámetros = {params}\n\n")
    print(f"\nResultados guardados en '{output_filename}'")
    
    best_params = top_results[0][0]
    print(f"\nMejores parámetros encontrados: {best_params}")


# %%
# 6. Visualización con los Mejores Parámetros
# --------------------------------------------------

if results:
    print("\nGenerando visualización con los mejores parámetros...")
    
    final_sample = test_images[0]['input'].unsqueeze(0)
    
    y_final = acquisition_model(final_sample)
    
    best_ista_solver = Ista(acquisition_model, fidelity, prior, **best_params)
    
    x0_final = acquisition_model.forward(y_final, type_calculation="backward")
    x_hat_final = best_ista_solver(y_final, x0=x0_final, gt=final_sample, verbose=True)

    basis = DCT2D()
    theta = basis.forward(x_hat_final).detach()
    normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    if acquisition_name == "spc":

        # 1. Calcular el tamaño del nuevo lienzo cuadrado
        side_length = int(math.sqrt(n_measurements)) + 1
        new_total_size = side_length * side_length     

        # 2. Crear un nuevo tensor (lienzo) lleno de ceros
        padded_y = torch.zeros(new_total_size, device=y_final.device)

        # 3. Copiar las mediciones originales al principio del lienzo
        padded_y[:n_measurements] = y_final.flatten()

        # 4. Ahora sí podemos remodelar el lienzo a un cuadrado 2D para visualización
        y_display = padded_y.reshape(1, 1, side_length, side_length)

    plt.figure(figsize=(16, 5))
    plt.suptitle(f"Reconstrucción con Mejores Parámetros ISTA (PSNR: {top_results[0][1]:.2f} dB)", fontsize=16)

    plt.subplot(1, 4, 1)
    plt.title("Referencia (GT)")
    plt.imshow(final_sample.cpu().squeeze().numpy(), cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 2)
    plt.title("Representación Esparsa (DCT)")
    plt.imshow(abs(normalize(theta.cpu()).squeeze().numpy()), cmap="gray")
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 4, 3)
    plt.title("Medición")
    plt.imshow(normalize(y_display.cpu()).squeeze().numpy(), cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 4)
    plt.title("Reconstrucción")
    plt.imshow(normalize(x_hat_final.cpu()).squeeze().detach().numpy(), cmap="gray")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()