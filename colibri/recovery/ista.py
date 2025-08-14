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


    def forward(self, y, gt=None, x0=None, verbose=False):
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
        for i in range(self.max_iters):

            x_old = x.clone()

            # gradient step
            x = x_old - self.alpha * self.fidelity.grad(x_old, y, self.H) 

            # proximal step
            x = self.prior.prox(x, self._lambda)
            
            error = self.fidelity.forward(x, y, self.H).item()
            errors.append(error)
            if gt is not None:
                psnrs.append(psnr(gt, x_old).item())
                mses.append(mse(gt, x_old).item())
                   
        # Graficar y guardar el error
        np.save('metricas/Ista_error.npy', errors)

        if gt is not None:
            np.save('metricas/Ista_psnr.npy', psnrs)
            np.save('metricas/Fista_mse.npy', mses)

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

        return x