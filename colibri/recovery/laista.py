import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity
from src.model import Accelerator # Asumiendo que esta es la ruta correcta
from colibri.metrics import psnr # Necesitarás importar la función PSNR

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
        if x0 is None:
            x0 = torch.zeros_like(y)
        
        x = x0
        z = x.clone()
        
        errors = []
        psnrs = []

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
                x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
                psnr_val = psnr(y_true=gt, y_pred=x_norm, data_range=1.0).item()
                psnrs.append(psnr_val)

        # Guardar métricas en archivos
        np.save('metricas/Laista_error.npy', errors)
        if gt is not None:
            np.save('metricas/Laista_psnr.npy', psnrs)

        # --- Visualización de resultados ---
        if verbose:
            if gt is not None:
                print(f'PSNR final: {psnrs[-1]:.2f} dB')
            
            # Gráfica del Error de Fidelidad
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(errors, color='b', label='LAISTA Fidelity')
            plt.yscale('log')
            plt.title('Convergencia del Error de Fidelidad')
            plt.ylabel(r'$\frac{1}{2} \|\mathbf{y} - \mathbf{H(x)}\|^2_2$', fontsize=14)
            plt.xlabel(r'Iteración', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            
            if gt is not None:
                # Gráfica del PSNR
                plt.subplot(1, 2, 2)
                plt.plot(psnrs, color='g', label='LAISTA PSNR')
                plt.title('Evolución del PSNR')
                plt.ylabel(r'PSNR (dB)', fontsize=14)
                plt.xlabel(r'Iteración', fontsize=14)
                plt.grid(True)
                plt.legend(fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
        return x