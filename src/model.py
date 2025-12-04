import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm

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
            self.up = spectral_norm(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class Decoder(nn.Module):
    def __init__(self, n_channels, bilinear=False, scaling:int=1):
        super(Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        # self.up1 = (Up(256 * scaling, 128 * scaling // factor, bilinear))
        self.up2 = (Up(128 * scaling, 64 * scaling // factor, bilinear))
        self.up3 = (Up(64 * scaling, 32 * scaling // factor, bilinear))
        self.up4 = (Up(32 * scaling, 16 * scaling, bilinear))
        self.outc = (OutConv(16 * scaling, n_channels))

    def forward(self, x):
        # x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)
    

class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=False, scaling:int=1):
        super(Encoder, self).__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16 * scaling))
        self.down1 = (Down(16 * scaling, 32 * scaling))
        self.down2 = (Down(32 * scaling, 64 * scaling))
        factor = 2 if bilinear else 1
        self.down3 = (Down(64 * scaling, 128 * scaling // factor))
        # factor = 2 if bilinear else 1
        # self.down4 = (Down(128 * scaling, 256 * scaling // factor))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        return x4

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)),
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

class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = spectral_norm(nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim))
        self.pw1 = spectral_norm(nn.Conv2d(dim, 4*dim, kernel_size=1))
        self.act = nn.GELU()
        self.pw2 = spectral_norm(nn.Conv2d(4*dim, dim, kernel_size=1))
        self.res_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return identity + self.res_scale * x


class EncoderConvNeXtLite(nn.Module):
    """
    Encoder estable, sin downsampling.
    """
    def __init__(self, in_channels=1, features=32, depth=3):
        super().__init__()

        self.stem = spectral_norm(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        )

        blocks = []
        for _ in range(depth):
            blocks.append(ConvNeXtLiteBlock(features))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

class DecoderConvNeXtLite(nn.Module):
    def __init__(self, out_channels=1, features=32, depth=3):
        super().__init__()

        blocks = []
        for _ in range(depth):
            blocks.append(ConvNeXtLiteBlock(features))

        self.blocks = nn.Sequential(*blocks)

        self.final = spectral_norm(
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.final(x)
        return x

class Accelerator(nn.Module):
    # def __init__(self, num_iterations, n_channels, bilinear=False, scaling:int=1):
    def __init__(self, num_iterations, decoder, encoder):
        super(Accelerator, self).__init__()

        # self.decoder = Decoder(n_channels, bilinear, scaling)
        # self.encoder_shared = Encoder(n_channels, bilinear, scaling)
        self.decoder = decoder
        self.encoder = encoder
        # self.encoders = nn.ModuleList([Encoder(n_channels, bilinear, scaling) for _ in range(num_iterations+1)])
        self.T = num_iterations

    def forward(self, x, history):

        #h = self.encoders[0](x)

        # for i in range(self.T):
            # h_i = self.encoders[i + 1](history[i]) 
            # h = h + h_i 
        h = self.encoder(x)

        for h_prev in history:
            h_i = self.encoder(h_prev)
            h = h + h_i

        h = h / (self.T+1)

        v = self.decoder(h)

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
    def __init__(self, acquistion_model, fidelity=L2(), prior=Sparsity("dct"), max_iters=5, alpha=1e-3, _lambda=0.1, num_iterations=3, n_channels=1, device = None):
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
        # self.alpha = alpha
        # self._lambda = _lambda
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self._lambda = nn.Parameter(torch.tensor(_lambda, dtype=torch.float32))

        # Red neuronal de aceleración y sus parámetros
        self.T = num_iterations
        self.n_channels = n_channels
        self.device = device
        decoder = Decoder(self.n_channels, False, 1)
        encoder = Encoder(self.n_channels, False, 1)
        # encoder = EncoderConvNeXtLite(in_channels=1, features=32, depth=3)
        # decoder = DecoderConvNeXtLite(out_channels=1, features=32, depth=3)
        self.acc = Accelerator(
            num_iterations= self.T,
            decoder= decoder,
            encoder= encoder
            ).to(device)
        self.norm = lambda x: torch.linalg.norm(x.flatten(start_dim=1), ord=2, dim=-1)

    def forward(self, y, gt=None, x0=None, ratio=None, verbose=False, return_rates_matrix=False):
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
        
        if x0 is None:
            x0 = torch.zeros_like(y)
        
        x = x0
        z = x.clone()
        
        # initial_x = x.detach().clone() # Usamos una copia desatachada
        history = [x.clone()] * self.T 
        # grad = self.alpha*self.fidelity.grad(z, y, self.H)
        # history = [grad.detach().clone()] * self.T

        errors = []
        psnrs = []
        mses = []
        convergence_rates = []
        changes_vectors = []

        if gt is not None:
            # Usamos torch.linalg.norm() para la norma Euclidiana (L2)
            error_prev = self.norm(x - gt)
            # --------------------------------------

        for i in range(self.max_iters):
            x_old = x.clone()
            # Paso de gradiente y proximal (actualización de x)
            # x.requires_grad_(True)
            # grad = self.fidelity.grad(z, y, self.H)
            # print(f'shape of grad {grad.shape} in iteration {i}')
            x = z - self.alpha * self.fidelity.grad(z, y, self.H)
            # x = z - self.alpha*grad
            x = self.prior.prox(x, self._lambda)
            x.requires_grad_(True)
            # Paso de aceleración aprendido (actualización de z)
            z = x + self.acc(x_old, history)
            # z = x + self.acc(grad, history)
            # z = self.prior.prox(z, self._lambda)
            # z.requires_grad_(True)
            
            history.append(x)
            # history.append(grad.detach().clone())
            if len(history) > self.T:
                # print('entra')
                history.pop(0)

            x_detached = x.detach()

            

            # --- Cálculo y almacenamiento de métricas ---

            # --- CÁLCULO DE LA TASA DE CONVERGENCIA ---
            if gt is not None:
                # Error actual (numerador): ||x_l - x*||
                error_curr = self.norm(x - gt) 
                # Tasa de convergencia: r(l) = ||x_l - x*|| / ||x_{l-1} - x*||
                # Evitamos la división por cero si el error_prev es 0.
                rate = (error_curr / error_prev)
                convergence_rates.append(rate.cpu())
                # Actualizar el error anterior para la próxima iteración
                error_prev = error_curr
                # ------------------------------------------
            
            if gt is not None and return_rates_matrix==False:
                # Normalizar la reconstrucción para un cálculo de PSNR correcto
                error = self.fidelity.forward(x, y, self.H).detach()
                errors.append(error.cpu())
                if x_detached.max() > x_detached.min():
                    x_norm = (x_detached - x_detached.min()) / (x_detached.max() - x_detached.min())
                else:
                    x_norm = x_detached
                psnrs.append(psnr(gt, x_norm, data_range=1.0).cpu())
                mses.append(mse(gt, x_norm).cpu())
                changes_vectors.append(self.norm(x-x_old).cpu())

        # Guardar métricas en archivos
        if gt is not None and return_rates_matrix==False:
            if ratio is not None:
                np.save(f'metricas/Laista_psnr{ratio}.npy', psnrs)
                np.save(f'metricas/Laista_mse{ratio}.npy', mses)
                np.save(f'metricas/Laista_convergence_rate{ratio}.npy', convergence_rates)
                np.save(f'metricas/Laista_change_vector{ratio}.npy', changes_vectors)                 
            else:
                np.save(f'metricas/Laista_psnr.npy', psnrs)
                np.save(f'metricas/Laista_mse.npy', mses)
                np.save(f'metricas/Laista_convergence_rate.npy', convergence_rates)
                np.save(f'metricas/Laista_change_vector.npy', changes_vectors)    

        # --- Visualización de resultados ---
        if verbose:
            if gt is not None:
                print(f'PSNR 1 sample: {np.array(psnrs)[:,0][-1]}')
                print(f'MSE 1 sample: {np.array(mses)[:,0][-1]}')
            
            # Gráfica del Error de Fidelidad
            plt.figure(figsize=(12, 5))
            plt.plot(np.array([e.detach().cpu().numpy() for e in errors]), color='b', label='LAISTA Fidelity')
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
        stacked_rates = torch.stack(convergence_rates)
        R_LAISTA_matrix = stacked_rates.T.contiguous()
        R_LAISTA_matrix = R_LAISTA_matrix.to(self.device)
        if return_rates_matrix == True:
            return z, R_LAISTA_matrix
        else:
            return z   
