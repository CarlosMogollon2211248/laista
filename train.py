import os
import random
import yaml

import numpy as np
import torch
import torch.nn as nn
import wandb

# --- Modulos del proyecto ---
from src.dataset import get_dataloaders
from src.model import Laista
from src.train_test import evaluate, train_one_epoch
from src.utils import get_hadamard_patterns  # Importamos la función para los patrones

# --- Modulos de colibri ---
from colibri.optics import SPC
from colibri.recovery.terms.prior import Sparsity, Denoiser
from colibri.recovery.terms.fidelity import L2

wandb.login(key="e12adcce380e93cac31fbde78d8e8d3b8fb94a90")

def set_seed(seed):
    """Fija las semillas de aleatoriedad para que los experimentos sean reproducibles."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Estas dos últimas líneas aseguran un comportamiento determinista en CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config_path='configs/spc_fashionmnist.yaml'):
    """Función principal para orquestar el entrenamiento."""
    
    # 1. Cargar la Configuración desde el archivo .yaml
    # ----------------------------------------------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Configuración del Entorno
    # ----------------------------------
    set_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # 3. Preparar los Datos
    # -----------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['data']['batch_size'],
        img_size=config['data']['img_size']
    )

    # 4. Preparar Modelos y Optimizador
    # --------------------------------------
    # A. Modelo de adquisición (SPC con patrones de Hadamard)
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

    # Congelar los parametros del denoiser
    for param in prior.parameters():
        param.requires_grad = False

    model = Laista(
        acquistion_model=acquisition_model,
        fidelity = fidelity,
        prior = prior,
        **config['laista_params'],
        **config['net_params']
    ).to(device)

    # Optimizador y Función de Pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = nn.MSELoss()

    # <--- NUEVO: Inicialización del Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # 5. Inicializar Logging (Weights & Biases)
    # ---------------------------------------------
    wandb.init(
        project=config['wandb']['project'], 
        name=config['wandb']['name'], 
        config=config
    )
    wandb.watch(model, log='all', log_freq=100) 

    # 6. Bucle de Entrenamiento y Validación
    # --------------------------------------
    best_val_loss = float('inf')
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoints_dir, f"best_model_{config['wandb']['name']}.pth")

    for epoch in range(config['training']['max_epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['max_epochs']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss, 'learning_rate': current_lr})
        
        # Guardar el mejor modelo hasta ahora (basado en la pérdida de validación)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
    
            torch.save(checkpoint, best_model_path)
            print(f"--> Nuevo mejor modelo guardado {best_model_path}")
            print(f"--> Nuevo mejor checkpoint guardado en la época {epoch + 1}")

    # 7. Evaluación Final en el Test Set
    # --------------------------------------
    print("\n--- Entrenamiento finalizado. Evaluando en el Test Set con el mejor modelo. ---")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = evaluate(model, test_loader, loss_fn, device)
    
    print(f"\n===================================================")
    print(f"RESULTADO FINAL - Test Loss: {test_loss:.6f}")
    print(f"===================================================")
    wandb.log({'final_test_loss': test_loss})

    wandb.finish()


if __name__ == '__main__':
    main()