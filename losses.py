import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Inspired from https://github.com/ChristophReich1996/Mode_Collapse/blob/master/loss.py and https://github.com/tariktemur/RGAN-and-RaGAN-PyTorch/blob/master/utils.py
"""

def gp_lossf(disc, real_samples, fake_samples, lambda_gp=2.):
    alpha = torch.rand((real_samples.shape[0], 1), device=real_samples.device)
    samples_interp = (alpha * real_samples + (1. - alpha) * fake_samples)
    samples_interp.requires_grad = True
    disc_pred_interp = disc(samples_interp)
    grads = torch.autograd.grad(outputs=disc_pred_interp.sum(),
                                inputs=samples_interp,
                                create_graph=True,
                                retain_graph=True)[0]
    gradient_penalty = (grads.view(grads.shape[0], -1).norm(dim=1) - 1.).pow(2).mean()
    return gradient_penalty * lambda_gp

def get_lossf(name):
    match name:
        case 'sgan':
            return GANGenLoss(), GANDiscLoss()
        case 'nsgan':
            return NSGANGenLoss(), NSGANDiscLoss()
        case 'wgan':
            return WGANGenLoss(), WGANDiscLoss()
        case 'lsgan':
            return LSGANGenLoss(), LSGANDiscLoss()
        case 'hinge':
            return HingeGANGenLoss(), HingeGANDiscLoss()
        case 'ralsgan':
            return RALSGANGenLoss(), RALSGANDiscLoss()
        case 'rahinge':
            return RAHingeGANGenLoss(), RAHingeGANDiscLoss()
        case _:
            raise NotImplementedError(f"Loss function {name} not implemented yet!")
        
class GANGenLoss(nn.Module):
    """
    Standard GAN Loss for generator
    """
    def forward(self, disc_pred_fake, **kwargs):
        return -F.softplus(disc_pred_fake).mean()
    
class GANDiscLoss(nn.Module):
    """
    Standard GAN Loss for discriminator
    """
    optimal_val = 0.6
    def forward(self, disc_pred_real, disc_pred_fake, **kwargs):
        return F.softplus(-disc_pred_real).mean() \
               + F.softplus(disc_pred_fake).mean()

class NSGANGenLoss(nn.Module):
    """
    Non-saturating GAN Loss for generator
    """
    optimal_val = 0.6
    def forward(self, disc_pred_fake, **kwargs):
        return F.softplus(-disc_pred_fake).mean()
    
class NSGANDiscLoss(GANDiscLoss):
    """
    Non-saturating GAN Loss for discriminator.
    """

class WGANGenLoss(nn.Module):
    """
    Wasserstein GAN Loss for generator
    """
    def forward(self, disc_pred_fake, **kwargs):
        return -disc_pred_fake.mean()
    
class WGANDiscLoss(nn.Module):
    """
    Wasserstein GAN Loss for discriminator
    """
    optimal_val = 0
    def forward(self, disc_pred_real, disc_pred_fake, **kwargs):
        return -disc_pred_real.mean() \
               + disc_pred_fake.mean()

class LSGANGenLoss(nn.Module):
    """
    Least Squares GAN Loss for generator
    """
    def forward(self, disc_pred_fake, **kwargs):
        return -0.5 * (disc_pred_fake - 1.).pow(2).mean()
    
class LSGANDiscLoss(nn.Module):
    """
    Least Squares GAN Loss for discriminator
    """
    optimal_val = 0.5
    def forward(self, disc_pred_real, disc_pred_fake, **kwargs):
        return 0.5 * ((-disc_pred_real - 1.).pow(2).mean()
                      + disc_pred_fake.pow(2).mean())
    
class HingeGANGenLoss(WGANGenLoss):
    """
    Hinge GAN Loss for generator
    """
    
class HingeGANDiscLoss(nn.Module):
    """
    Hinge GAN Loss for discriminator
    """
    optimal_val = 0
    def forward(self, disc_pred_real, disc_pred_fake, **kwargs):
        return - F.relu(disc_pred_real - 1.).mean() \
               - F.relu(-disc_pred_fake - 1.).mean()

class RALSGANGenLoss(nn.Module):
    """
    Relativistic Average Least Squares GAN Loss for generator
    """
    def forward(self, disc_pred_fake, disc, real_samples, **kwargs):
        disc_pred_real = disc(real_samples)
        fake_mean = torch.mean(disc_pred_fake)
        real_mean = torch.mean(disc_pred_real)
        ones = torch.ones_like(disc_pred_fake)
        return F.mse_loss(disc_pred_fake - fake_mean, ones) + F.mse_loss(disc_pred_real - real_mean, -ones)
    
class RALSGANDiscLoss(nn.Module):
    """
    Relativistic Average Least Squares GAN Loss for discriminator
    """
    optimal_val = 1
    def forward(self, disc_pred_real, disc_pred_fake, **kwargs):
        fake_mean = torch.mean(disc_pred_fake)
        real_mean = torch.mean(disc_pred_real)
        ones = torch.ones_like(disc_pred_fake)
        return F.mse_loss(disc_pred_real - real_mean, ones) + F.mse_loss(disc_pred_fake - fake_mean, -ones)
    
class RAHingeGANGenLoss(nn.Module):
    """
    Relativistic Average Hinge GAN Loss for generator
    """
    def forward(self, disc_pred_fake, disc, real_samples, **kwargs):
        disc_pred_real = disc(real_samples)
        fake_mean = torch.mean(disc_pred_fake)
        real_mean = torch.mean(disc_pred_real)
        ones = torch.ones_like(disc_pred_fake)
        diff_r_f = disc_pred_real - real_mean
        diff_f_r = disc_pred_fake - fake_mean
        return torch.clamp((ones + diff_r_f), min=0).mean() + torch.clamp((ones - diff_f_r), min=0).mean()
    
class RAHingeGANDiscLoss(nn.Module):
    """
    Relativistic Average Hinge GAN Loss for discriminator
    """
    optimal_val = 2
    def forward(self, disc_pred_real, disc_pred_fake, **kwargs):
        fake_mean = torch.mean(disc_pred_fake)
        real_mean = torch.mean(disc_pred_real)
        ones = torch.ones_like(disc_pred_fake)
        diff_r_f = disc_pred_real - real_mean
        diff_f_r = disc_pred_fake - fake_mean
        return torch.clamp((ones - diff_r_f), min=0).mean() + torch.clamp((ones + diff_f_r), min=0).mean()
