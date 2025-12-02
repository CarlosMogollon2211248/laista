import torch
from torch import nn

from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity
import matplotlib.pyplot as plt
import numpy as np
from colibri.metrics import psnr, mse

class Ista(nn.Module):
    r"""
    Iterative Shrinkage-Thresholding Algorithm (ISTA)

    The ISTA algorithm solves the optimization problem:

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{arg min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The ISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.
    """

    def __init__(self, acquistion_model, fidelity=L2(), prior=Sparsity("dct"), max_iters=5, alpha=1e-3, _lambda=0.1):
        r"""
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the FISTA algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.
            _lambda (float): The regularization parameter for the prior term. Defaults to 0.1.

        Returns:
            None
        """
        super(Ista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior

        self.H = lambda x: self.acquistion_model.forward(x)

        self.max_iters = max_iters
        self.alpha = alpha
        self._lambda = _lambda
        self.norm = lambda x: torch.linalg.norm(x.flatten(start_dim=1), ord=2, dim=-1)

    def forward(self, y, gt=None, x0=None, verbose=False, ratio=None):
        r"""Runs the ISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        errors = []
        psnrs = []
        mses = []
        convergence_rates = []

        if gt is not None:
            # Usamos torch.linalg.norm() para la norma Euclidiana (L2)
            error_prev = self.norm(x - gt)
            # --------------------------------------

        for i in range(self.max_iters):

            x_old = x.clone()

            # gradient step
            x = x_old - self.alpha * self.fidelity.grad(x_old, y, self.H) 

            # proximal step
            x = self.prior.prox(x, self._lambda)

            # --- CÁLCULO DE LA TASA DE CONVERGENCIA ---
            if gt is not None:
                # Error actual (numerador): ||x_l - x*||
                error_curr = self.norm(x - gt) 
        
                # Tasa de convergencia: r(l) = ||x_l - x*|| / ||x_{l-1} - x*||
                # Evitamos la división por cero si el error_prev es 0.
                if error_prev.item() != 0:
                    rate = (error_curr / error_prev).item()
                    convergence_rates.append(rate)
                else:
                    # Si el error anterior era 0, el modelo ya convergió
                    convergence_rates.append(0.0) 

                # Actualizar el error anterior para la próxima iteración
                error_prev = error_curr
                # ------------------------------------------

            error = self.fidelity.forward(x, y, self.H).item()
            errors.append(error)
            if gt is not None:
                psnrs.append(psnr(gt, x_old).item())
                mses.append(mse(gt, x_old).item())
                   
        # Graficar y guardar el error
        np.save('metricas/Ista_error.npy', errors)

        if gt is not None:
            if ratio is not None:
                np.save(f'metricas/Ista_psnr{ratio}.npy', psnrs)
                np.save(f'metricas/Ista_mse{ratio}.npy', mses)
                np.save(f'metricas/Ista_convergence_rate{ratio}.npy', convergence_rates) 
            else:
                np.save(f'metricas/Ista_psnr.npy', psnrs)
                np.save(f'metricas/Ista_mse.npy', mses)
                np.save(f'metricas/Ista_convergence_rate.npy', convergence_rates) 

        if verbose:
            if gt is not None:
                print(f'PSNR: {psnrs[-1]}')
                print(f'MSE: {mses[-1]}')
            plt.figure()
            plt.plot(errors, color = 'r', label = 'ISTA Fidelity')
            plt.yscale('log')
            plt.ylabel(r'$\frac{1}{2} \|\mathbf{y} - \mathbf{H(x)}\|^2$', fontsize=14)
            plt.xlabel(r'Iteration', fontsize=14)
            plt.grid('on')
            plt.legend(fontsize=14)
            if gt is not None:
                plt.figure()
                plt.plot(psnrs, color = 'r', label = 'ISTA psnr')
                plt.ylabel(r'PSNR (dB)', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)

                plt.figure()
                plt.plot(mses, color = 'b', label = 'ISTA MSE')
                plt.yscale('log') # El MSE a menudo se ve mejor en escala logarítmica
                plt.ylabel(r'MSE', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)

                plt.figure()
                plt.plot(convergence_rates, color = 'g', label = 'ISTA Convergence Rate')
                plt.ylabel(r'$r(l)$', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)  

        return x