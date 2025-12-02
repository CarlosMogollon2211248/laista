import os
import random
import yaml
import numpy as np
import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader, ConcatDataset

# --- IMPORTACIONES ---
# Asegúrate de que todas estas importaciones sean válidas en tu entorno
from src.dataset import get_dataloaders # Para obtener los subsets del dataset
# Importar módulos del demo:
from colibri.optics import SPC, SD_CASSI, DD_CASSI, C_CASSI
from colibri.recovery.fista import Fista
from colibri.recovery.terms.prior import Denoiser 
from colibri.recovery.terms.fidelity import L2
from libs.ordering import get_index_matrix
from libs.ordering.sequency import sequency_order
from libs.row_wise import hadamard_row

# --- Funciones de Ayuda (Copiadas del demo) ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_hadamard_patterns(size, n_measurements, n=10, ordering="cake_cutting"):
    # Función para replicar la lógica de CA Generado del demo
    size = np.sqrt(2**n).astype(int)
    M, N = size, size
    
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
    return H[:n_measurements, :M*N]

# --- FUNCIÓN PRINCIPAL DE GENERACIÓN ---

def generate_reference_rates():
    
    # 1. Configuración
    config_path = 'configs/spc_fashionmnist.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. Obtener DataLoader Completo (SIN SHUFFLE)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['data']['batch_size'], 
        img_size=config['data']['img_size']
    )
    # Concatenar todos los subsets para asegurar las 10000 muestras en orden
    full_dataset_loader = DataLoader(
        ConcatDataset([train_loader.dataset, val_loader.dataset, test_loader.dataset]), 
        batch_size=config['data']['batch_size'], 
        shuffle=False 
    )
    
    # 3. Inicialización de Modelos (Copiado del demo)
    img_h, img_w = config['data']['img_size'], config['data']['img_size']
    ratio = 0.7 
    n_measurements = int(ratio * (img_h * img_w))
    
    initial_ca = generate_hadamard_patterns(img_h, n_measurements)
    
    acquisition_config = {
        'input_shape': tuple(config['acquisition']['input_shape']),
        'n_measurements': n_measurements,
        'initial_ca': initial_ca 
    }
    acquisition_model = SPC(**acquisition_config).to(device)

    # Parámetros FISTA (Copiados del demo)
    algo_params = {
        "max_iters": 180,
        "alpha": 0.0004774212882981862,
        "_lambda": 0.010969419598768213,
    }
    
    fidelity = L2()
    prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download_lipschitz", 'device': device}).to(device)
    
    fista_model = Fista(acquisition_model, fidelity, prior, **algo_params).to(device)
    
    # 4. BUCLE DE GENERACIÓN
    all_rates_tensors = []
    
    print("Iniciando generación de R* (10000 muestras x 180 iteraciones)...")
    
    for data in tqdm(full_dataset_loader, desc="Generando R*", colour='magenta'):
        img_gt = data[0].to(device)
            
        # Medición y x0
        y = acquisition_model(img_gt)
        x0 = acquisition_model.forward(y, type_calculation="backward")
            
        # Ejecutar FISTA y obtener la matriz de tasas
        # Asumiendo que fista_model.forward(..., return_rates_matrix=True) devuelve el tensor [B, Iters]
        _, R_batch_matrix = fista_model(y, gt=img_gt, x0=x0, return_rates_matrix=True) 
            
        all_rates_tensors.append(R_batch_matrix)

    # 5. GUARDADO FINAL
    
    R_star_full = np.concatenate(all_rates_tensors, axis=1) # Shape [10000, 180]
    R_star_full = R_star_full.T
    output_path = 'FISTA_optimal_convergence_R_star.npy'
    np.save(output_path, R_star_full)
    
    print("\n" + "="*50)
    print("✅ GENERACIÓN FINALIZADA")
    print(f"Dataset de Tasa de Convergencia (R*) guardado en: {output_path}")
    print(f"Shape FINAL del Dataset R*: {R_star_full.shape}")
    print("="*50)


if __name__ == '__main__':
    generate_reference_rates()