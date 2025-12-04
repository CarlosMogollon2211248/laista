import os
import random
import yaml

import numpy as np
import torch
import torch.nn as nn
import wandb

# --- Modulos del proyecto ---
from src.dataset import get_dataloaders, get_convergence_dataloaders
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
    LAMBDA_CONVERGENCE = 0.1
    # 3. Preparar los Datos
    # -----------------------
    # train_loader, val_loader, test_loader = get_dataloaders(
    #     batch_size=config['data']['batch_size'],
    #     img_size=config['data']['img_size']
    # )
    # train_loader, val_loader, test_loader = get_convergence_dataloaders(
    #     config, 
    #     R_star_path='FISTA_optimal_convergence_R_star.npy' # Ajusta la ruta si es necesario
    #     )
    train_loader, val_loader, test_loader = get_convergence_dataloaders(
        config, 
        R_star_path='FISTA_optimal_change_vector.npy' # Ajusta la ruta si es necesario
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
    # prior = Sparsity(basis="dct")
    prior = Denoiser({'in_channels': 1, 'out_channels': 1, 'pretrained': "download_lipschitz", 'device': device}).to(device)
    # Congelar o descongelar los parametros del denoiser
    for param in prior.parameters():
        param.requires_grad = True

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, param in prior.named_parameters():
        # p.numel() obtiene el número total de elementos (parámetros) en el tensor.
        num_param = param.numel()
        total_params += num_param

        if param.requires_grad:
            trainable_params += num_param

        else:
            non_trainable_params += num_param

    print("--- Desglose de Parámetros del Denoiser ---")
    print(f"**Total de Parámetros:** {total_params:,}")
    print(f"**Parámetros Entrenables:** {trainable_params:,}")
    print(f"**Parámetros No Entrenables (Congelados):** {non_trainable_params:,}")
    print(f"**Porcentaje Entrenable:** {trainable_params / total_params * 100:.2f}%")


    model = Laista(
        acquistion_model=acquisition_model,
        fidelity = fidelity,
        prior = prior,
        **config['laista_params'],
        **config['net_params'],
        device= device
    ).to(device)

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, param in model.named_parameters():
        # p.numel() obtiene el número total de elementos (parámetros) en el tensor.
        num_param = param.numel()
        total_params += num_param

        if param.requires_grad:
            trainable_params += num_param

        else:
            non_trainable_params += num_param

    print("--- Desglose de Parámetros del Modelo ---")
    print(f"**Total de Parámetros:** {total_params:,}")
    print(f"**Parámetros Entrenables:** {trainable_params:,}")
    print(f"**Parámetros No Entrenables :** {non_trainable_params:,}")
    print(f"**Porcentaje Entrenable:** {trainable_params / total_params * 100:.2f}%")

    # Optimizador y Función de Pérdida
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'] 
        )
    loss_fn = nn.MSELoss(reduction='none')

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=5
    # )
    # Inicialización del Scheduler (Ya la tienes bien)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=5 
    #     )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1000001)

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
        
        train_loss, train_psnr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, LAMBDA_CONVERGENCE)
        val_loss, val_psnr = evaluate(model, val_loader, loss_fn, device, LAMBDA_CONVERGENCE)
        
        # scheduler.step(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'learning_rate': current_lr, 
                   'train_psnr': train_psnr, 'val_psnr': val_psnr})
        
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
    # Sección 7. Evaluación Final en el Test Set
    # --------------------------------------
    print("\n--- Entrenamiento finalizado. Evaluando en el Test Set con el mejor modelo. ---")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # checkpoint = torch.load(best_model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # ¡CORRECCIÓN AQUÍ! Desempaquetar la tupla
    # sample = next(iter(test_loader))[0].to(device)[0:1]#.unsqueeze(0)
    # y = acquisition_model(sample,type_calculation='forward')
    # x0 = acquisition_model(y,type_calculation='backward')
    # x_hat = model(y, x0=x0, gt=sample, verbose=True)

    test_loss, test_psnr = evaluate(model, test_loader, loss_fn, device, LAMBDA_CONVERGENCE)

    print(f"\n===================================================")
    # Usar solo la variable test_loss (que ahora es el float)
    print(f"RESULTADO FINAL - Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.4f} dB")   
    print(f"===================================================")

    wandb.log({'final_test_loss': test_loss, 'final_test_psnr': test_psnr})

    wandb.finish()


if __name__ == '__main__':
    main()