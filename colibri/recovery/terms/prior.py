import torch

from .transforms import DCT2D


class Sparsity(torch.nn.Module):
    r"""
        Sparsity prior 
        
        .. math::
        
            g(\mathbf{x}) = \| \transform \textbf{x}\|_1
        
        where :math:`\transform` is the sparsity basis and :math:`\textbf{x}` is the input tensor.

    """
    def __init__(self, basis=None):
        r"""
        Args:
            basis (str): Basis function. 'dct', 'None'. Default is None.
        """
        super(Sparsity, self).__init__()

        if basis == 'dct':
            self.transform = DCT2D()
        else:
            self.transform = None

    def forward(self, x):
        r"""
        Compute sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Sparsity term.
        """
        x = self.transform.forward(x)
        return torch.norm(x, 1)**2
    
    def prox(self, x, _lambda, type="soft"):
        r"""
        Compute proximal operator of the sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
            _lambda (float): Regularization parameter.
            type (str): String, it can be "soft" or "hard".
        
        Returns:
            torch.Tensor: Proximal operator of the sparsity term.
        """
        
        x = x.requires_grad_()
        x = self.transform.forward(x)

        if type == 'soft':
            x = torch.sign(x)*torch.max(torch.abs(x) - _lambda, torch.zeros_like(x))
        elif type == 'hard':
            x = x*(torch.abs(x) > _lambda)
        
        x = self.transform.inverse(x)
        return x
        
    def transform(self, x):
        
        if self.transform is not None:
            return self.transform.forward(x)
        else:
            return x
    
    def inverse(self, x):
        
        if self.transform is not None:
            return self.transform.inverse(x)
        else:
            return x
        
from deepinv.models import DnCNN, Restormer, SCUNet, SwinIR, UNet, DRUNet

class Denoiser(torch.nn.Module):

    def __init__(self, denoiser_args=None,denoiser='DnCNN'):
        super().__init__()

        if denoiser == 'DnCNN':
            self.denoiser = DnCNN(**denoiser_args)
        if denoiser == 'Restormer':
            self.denoiser = Restormer(**denoiser_args)
            
        if denoiser == 'SCUNet':
            self.denoiser = SCUNet(**denoiser_args)
        if denoiser == 'SwinIR':
            self.denoiser = SwinIR(**denoiser_args)
        if denoiser == 'UNet':
            self.denoiser = UNet(**denoiser_args)
        if denoiser == 'DRUNet':
            self.denoiser = DRUNet(**denoiser_args)

    def prox(self, x, _lambda):
        with torch.no_grad():
            if x.shape[1] == 1:
                x = self.denoiser(x,_lambda)
            
            if x.shape[1] == 2:
                x_c1 = x[:, 0:1, :, :]  
                x_c2 = x[:, 1:2, :, :]

                d_c1 = self.denoiser(x_c1)
                d_c2 = self.denoiser(x_c2)

                x = torch.cat((d_c1, d_c2), dim=1)
        return x

    

