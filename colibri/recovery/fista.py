import torch
from torch import nn

from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity
import matplotlib.pyplot as plt
import numpy as np
from colibri.metrics import psnr, mse
from tqdm import tqdm

class Fista(nn.Module):
    r"""
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    The FISTA algorithm solves the optimization problem:

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{arg min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The FISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        t_{k+1} &= \frac{1 + (1 + 4t_k^2)^{0.5}}{2} \\
        \mathbf{z}_{k+1} &=  \mathbf{x}_{k+1} + \frac{t_k-1}{t_{k+1}}( \mathbf{x}_{k} - \mathbf{x}_{k-1})
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.

    Implementation based on the formulation of authors in https://doi.org/10.1137/080716542
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
        super(Fista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior

        self.H = lambda x: self.acquistion_model.forward(x)

        self.max_iters = max_iters
        self.alpha = alpha
        self._lambda = _lambda
        self.norm = lambda x: torch.linalg.norm(x.flatten(start_dim=1), ord=2, dim=-1)

    def forward(self, y, gt=None, x0=None, verbose=False, ratio=None, return_rates_matrix=False):
        r"""Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.
            gt: Ground Truth: Si no se requiere el PSNR colocar gt en None

        Returns:
            torch.Tensor: The reconstructed image.
        """
        
        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        t = 1
        z = x.clone()
        errors = []
        psnrs = []
        mses = []
        convergence_rates = []
        changes_vectors = []

        if gt is not None:
            # Usamos torch.linalg.norm() para la norma Euclidiana (L2)
            error_prev = self.norm(x - gt)
            # --------------------------------------

        for i in tqdm(range(self.max_iters), colour='green'):
            x_old = x.clone()

            # gradient step
            x = z - self.alpha * self.fidelity.grad(z, y, self.H) 

            # proximal step
            x = self.prior.prox(x, self._lambda)

            # FISTA step
            t_old = t
            t = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)

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

            error = self.fidelity.forward(x, y, self.H)
            errors.append(error.cpu())
            if gt is not None:
                psnrs.append(psnr(gt, x_old, data_range=1.0).cpu())
                mses.append(mse(gt, x_old).cpu())
                changes_vectors.append(self.norm(x-x_old).cpu())

        # Graficar y guardar el error
        np.save('metricas/Fista_error.npy', errors)

        if gt is not None:
            if ratio is not None:
                np.save(f'metricas/Fista_error{ratio}.npy', errors)
                np.save(f'metricas/Fista_psnr{ratio}.npy', psnrs)
                np.save(f'metricas/Fista_mse{ratio}.npy', mses)
                np.save(f'metricas/Fista_convergence_rate{ratio}.npy', convergence_rates) 
                np.save(f'metricas/Fista_change_vector{ratio}.npy', changes_vectors) 
            else:
                np.save(f'metricas/Fista_error.npy', errors)                
                np.save(f'metricas/Fista_psnr.npy', psnrs)
                np.save(f'metricas/Fista_mse.npy', mses)
                np.save(f'metricas/Fista_convergence_rate.npy', convergence_rates) 
                np.save(f'metricas/Fista_change_vector.npy', changes_vectors) 

        if verbose:
            if gt is not None:
                print(f'PSNR 1 sample: {np.array(psnrs)[:,0][-1]}')
                print(f'MSE 1 sample: {np.array(mses)[:,0][-1]}')
            plt.figure()
            plt.plot(errors, color = 'r', label = 'FISTA Fidelity')
            plt.yscale('log')
            plt.ylabel(r'$\frac{1}{2} \|\mathbf{y} - \mathbf{H(x)}\|^2$', fontsize=14)
            plt.xlabel(r'Iteration', fontsize=14)
            plt.grid('on')
            plt.legend(fontsize=14)
            if gt is not None:
                plt.figure()
                plt.plot(psnrs, color = 'r', label = 'FISTA psnr')
                plt.ylabel(r'PSNR (dB)', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)

                plt.figure()
                plt.plot(mses, color = 'b', label = 'FISTA MSE')
                plt.yscale('log') # El MSE a menudo se ve mejor en escala logarítmica
                plt.ylabel(r'MSE', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)

                plt.figure()
                plt.plot(convergence_rates, color = 'g', label = 'FISTA Convergence Rate')
                plt.ylabel(r'$r(l)$', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)

                plt.figure()
                plt.plot(changes_vectors, color = 'g', label = 'FISTA Change Vector')
                plt.ylabel(r'$r(l)$', fontsize=14)
                plt.xlabel(r'Iteration', fontsize=14)
                plt.grid('on')
                plt.legend(fontsize=14)                 
        if return_rates_matrix == True:
            return x, changes_vectors
        else:
            return x