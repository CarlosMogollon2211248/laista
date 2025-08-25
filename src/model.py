import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity
from colibri.metrics import psnr, mse

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class Decoder(nn.Module):
    def __init__(self, n_channels, bilinear=False, scaling:int=1):
        super(Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = (Up(256 * scaling, 128 * scaling // factor, bilinear))
        self.up2 = (Up(128 * scaling, 64 * scaling // factor, bilinear))
        self.up3 = (Up(64 * scaling, 32 * scaling // factor, bilinear))
        self.up4 = (Up(32 * scaling, 16 * scaling, bilinear))
        self.outc = (OutConv(16 * scaling, n_channels))

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=False, scaling:int=1):
        super(Encoder, self).__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16 * scaling))
        self.down1 = (Down(16 * scaling, 32 * scaling))
        self.down2 = (Down(32 * scaling, 64 * scaling))
        self.down3 = (Down(64 * scaling, 128 * scaling))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128 * scaling, 256 * scaling // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Accelerator(nn.Module):
    def __init__(self, num_iterations, n_channels, bilinear=False, scaling:int=1):
        super(Accelerator, self).__init__()

        self.decoder = Decoder(n_channels, bilinear, scaling)
        self.encoders = nn.ModuleList([Encoder(n_channels, bilinear, scaling) for _ in range(num_iterations)])
        self.history = []

    def forward(self, x):
        # Siempre añade la reconstrucción más reciente
        self.history.append(x.detach())

        # Si el historial es demasiado largo, quita el elemento más antiguo
        if len(self.history) > len(self.encoders): # len(self.encoders) es tu T
            self.history.pop(0)

        
        h = 0
        for i in range(len(self.history)):
            h_i = self.encoders[i](self.history[i])
            h = h + h_i  
        h = h / len(self.history)

        v = self.decoder(h)
        v = v + self.history[-1] 
        return v

class Laista(nn.Module):
    r"""
    Learned Accelerated Iterative Shrinkage-Thresholding Algorithm (LAISTA).

    LAISTA busca resolver el mismo problema de optimización que ISTA, pero introduce
    un paso de aceleración aprendido a través de una red neuronal (Accelerator).
    
    El problema de optimización es:
    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{arg min}} \quad \frac{1}{2}||\mathbf{y} - \mathbf{H}(\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    Las iteraciones del algoritmo combinan un paso de gradiente, un paso proximal
    y un paso de aceleración aprendido:
    .. math::
        \begin{align*}
           \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
           \mathbf{z}_{k+1} &= \text{Accelerator}(\mathbf{x}_{k+1})
        \end{align*}
    """
    def __init__(self, acquistion_model, fidelity=L2(), prior=Sparsity("dct"), max_iters=5, alpha=1e-3, _lambda=0.1, num_iterations=3, n_channels=1):
        r"""
        Args:
            acquistion_model (nn.Module): El modelo de adquisición del sistema (operador H).
            fidelity (nn.Module): El término de fidelidad (p. ej., L2).
            prior (nn.Module): El término de regularización (p. ej., Sparsity).
            max_iters (int): Número máximo de iteraciones.
            alpha (float): Tamaño del paso del gradiente.
            _lambda (float): Parámetro de regularización para el término prior.
            num_iterations (int): Parámetro para la red 'Accelerator'.
            n_channels (int): Parámetro para la red 'Accelerator'.
        """
        super(Laista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior
        self.H = lambda x: self.acquistion_model.forward(x)

        # Hiperparámetros fijos
        self.max_iters = max_iters
        self.alpha = alpha
        self._lambda = _lambda

        # Red neuronal de aceleración y sus parámetros
        self.T = num_iterations
        self.n_channels = n_channels
        self.acc = Accelerator(self.T, self.n_channels)

    def forward(self, y, gt=None, x0=None, verbose=False):
        r"""
        Ejecuta el algoritmo LAISTA.

        Args:
            y (torch.Tensor): Las mediciones a reconstruir.
            gt (torch.Tensor, optional): La imagen original (ground truth) para calcular PSNR.
            x0 (torch.Tensor, optional): La estimación inicial para la solución.
            verbose (bool): Si es True, imprime y grafica las métricas de convergencia.

        Returns:
            torch.Tensor: La imagen reconstruida.
        """

        self.acc.history = []
        
        if x0 is None:
            x0 = torch.zeros_like(y)
        
        x = x0
        z = x.clone()
        
        errors = []
        psnrs = []
        mses = []
        for i in range(self.max_iters):
            # Paso de gradiente y proximal (actualización de x)
            x = z - self.alpha * self.fidelity.grad(z, y, self.H)
            x = self.prior.prox(x, self._lambda)
            
            # Paso de aceleración aprendido (actualización de z)
            z = self.acc(x)

            # --- Cálculo y almacenamiento de métricas (igual que en ISTA) ---
            error = self.fidelity.forward(x, y, self.H).item()
            errors.append(error)
            
            if gt is not None:
                # Normalizar la reconstrucción para un cálculo de PSNR correcto
                x_norm = torch.sigmoid(z)
                psnrs.append(psnr(gt, x_norm).item())
                mses.append(mse(gt, x_norm).item())

        # Guardar métricas en archivos
        np.save('metricas/Laista_error.npy', errors)
        if gt is not None:
            np.save('metricas/Laista_psnr.npy', psnrs)
            np.save('metricas/Laista_mse.npy', mses)
        # --- Visualización de resultados ---
        if verbose:
            if gt is not None:
                print(f'PSNR: {psnrs[-1]}')
                print(f'MSE: {mses[-1]}')
            
            # Gráfica del Error de Fidelidad
            plt.figure(figsize=(12, 5))
            plt.plot(errors, color='b', label='LAISTA Fidelity')
            plt.yscale('log')
            plt.ylabel(r'$\frac{1}{2} \|\mathbf{y} - \mathbf{H(x)}\|^2_2$', fontsize=14)
            plt.xlabel(r'Iteración', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            
            if gt is not None:
                # Gráfica del PSNR
                plt.figure()
                plt.plot(psnrs, color='g', label='LAISTA PSNR')
                plt.ylabel(r'PSNR (dB)', fontsize=14)
                plt.xlabel(r'Iteración', fontsize=14)
                plt.grid(True)
                plt.legend(fontsize=12)

                plt.figure()
                plt.plot(mses, color = 'b', label = 'LAISTA MSE')
                plt.yscale('log') # El MSE a menudo se ve mejor en escala logarítmica
                plt.ylabel(r'MSE', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)

            plt.tight_layout()
            plt.show()
            
        return torch.sigmoid(z)
