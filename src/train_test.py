# Contenido para: src/train_test.py

import torch
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

# --- Clase de Ayuda para promediar métricas ---
# (Más adelante puedes mover esto a src/utils.py)
class AverageMeter:
    """Computa y almacena el valor actual y el promedio."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# -----------------------------------------------

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Ejecuta una época completa de entrenamiento.
    """
    model.train()  # Pone el modelo en modo de entrenamiento
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # tqdm crea una barra de progreso para el bucle
    data_loop = tqdm(dataloader, total=len(dataloader), desc="Training")
    
    for data in data_loop:
        img_gt = data[0].to(device)
        
        # 1. Poner a cero los gradientes
        optimizer.zero_grad()
        
        # 2. Forward pass completo del modelo
        y = model.acquistion_model(img_gt).to(device)
        x0 = model.acquistion_model.forward(y, type_calculation="backward").to(device)
        img_hat = model(y, x0=x0).to(device)

        if img_hat.max() > img_hat.min():
            img_hat_norm = (img_hat - img_hat.min()) / (img_hat.max() - img_hat.min())
        else:
            img_hat_norm = img_hat # Evitar división por cero
        # 3. Calcular la pérdida
        loss = loss_fn(img_hat_norm, img_gt)
        psnr = psnr_metric(img_hat_norm, img_gt).item()

        # 4. Backpropagation (cálculo de gradientes)
        loss.backward()
        
        # 5. Actualizar los pesos del modelo
        optimizer.step()
        
        # Actualizar y mostrar métricas
        loss_meter.update(loss.item(), img_gt.size(0))
        psnr_meter.update(psnr, img_gt.size(0))
        data_loop.set_postfix({
            'avg_loss': f'{loss_meter.avg:.4f}',
            'avg_psnr': f'{psnr_meter.avg:.2f} dB'
            })

    return loss_meter.avg, psnr_meter.avg

def evaluate(model, dataloader, loss_fn, device):
    """
    Ejecuta una época completa de evaluación/validación.
    """
    model.eval()  # Pone el modelo en modo de evaluación
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    data_loop = tqdm(dataloader, total=len(dataloader), desc="Evaluating")
    
    for data in data_loop:
        img_gt = data[0].to(device)
        
        # Forward pass completo
        y = model.acquistion_model(img_gt).to(device)
        x0 = model.acquistion_model.forward(y, type_calculation="backward").to(device)
        img_hat = model(y, x0=x0).to(device)
        if img_hat.max() > img_hat.min():
            img_hat_norm = (img_hat - img_hat.min()) / (img_hat.max() - img_hat.min())
        else:
            img_hat_norm = img_hat # Evitar división por cero

        loss = loss_fn(img_hat_norm, img_gt)
        psnr = psnr_metric(img_hat_norm, img_gt).item()

        # Actualizar y mostrar métricas
        loss_meter.update(loss.item(), img_gt.size(0))
        psnr_meter.update(psnr, img_gt.size(0))
        data_loop.set_postfix({
            'avg_loss': f'{loss_meter.avg:.4f}',
            'avg_psnr': f'{psnr_meter.avg:.2f} dB'
            })
        
    return loss_meter.avg, psnr_meter.avg