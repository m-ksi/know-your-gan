import argparse
import yaml
import torch
import numpy as np
from data import get_dataset
from models import get_model
from losses import get_lossf, gp_lossf, zero_centered_gp_lossf
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from logger import Logger
import os

def cosine_decay_with_warmup(cur_step, base_value, total_steps, final_value=0.0, warmup_value=0.0, warmup_steps=0, hold_base_value_steps=0):
    decay = 0.5 * (1 + np.cos(np.pi * (cur_step - warmup_steps - hold_base_value_steps) / float(total_steps - warmup_steps - hold_base_value_steps)))
    cur_value = base_value + (1 - decay) * (final_value - base_value)
    if hold_base_value_steps > 0:
        cur_value = np.where(cur_step > warmup_steps + hold_base_value_steps, cur_value, base_value)
    if warmup_steps > 0:
        slope = (base_value - warmup_value) / warmup_steps
        warmup_v = slope * cur_step + warmup_value
        cur_value = np.where(cur_step < warmup_steps, warmup_v, cur_value)
    return float(np.where(cur_step > total_steps, final_value, cur_value))

def get_optimizer(net, conf):
    # Adam, RMSprop
    params = net.parameters()
    match conf['type']:
        case 'Adam':
            cl = torch.optim.Adam
        case 'RMSprop':
            cl = torch.optim.RMSprop
        case 'AdamW':
            cl = torch.optim.AdamW
    optim = cl(params, **conf['params'])
    return optim

def train(cfg):
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    experiment_str = f"{cfg.data['type']}-{cfg.net_d['type']}-{cfg.lossf}"
    if cfg.net_d['params']['use_spectral_norm']:
        experiment_str = experiment_str + "-sn"
    if cfg.use_gp:
        experiment_str = experiment_str + "-gp"
    if cfg.mind_the_gap:
        experiment_str = experiment_str + "-gap"
    if cfg.skip_confident_model:
        experiment_str = experiment_str + "-skip"
    if cfg.use_r1_gp:
        experiment_str = experiment_str + "-r1"
    if cfg.use_r2_gp:
        experiment_str = experiment_str + "-r2"
    logger = Logger(f'experiments/{experiment_str}')

    dataset = get_dataset(cfg.data)
    dataloader = DataLoader(dataset, shuffle=True, **cfg.data['loader'])
    train_iter = iter(dataloader)
    batch_size = cfg.data['loader']['batch_size']
    latent_size = cfg.net_g['params']['nz']

    net_g = get_model(cfg.net_g).to(cfg.device)
    net_d = get_model(cfg.net_d).to(cfg.device)

    optim_g = get_optimizer(net_g, cfg.optim_g)
    optim_d = get_optimizer(net_d, cfg.optim_d)

    if cfg.resume_from:
        resume_step = int(cfg.resume_from.split('/')[-1])
        net_g.load_state_dict(torch.load(f"{cfg.resume_from}/net_g.state_dict.pth"))
        net_d.load_state_dict(torch.load(f"{cfg.resume_from}/net_d.state_dict.pth"))
        optim_g.load_state_dict(torch.load(f"{cfg.resume_from}/optim_g.state_dict.pth"))
        optim_d.load_state_dict(torch.load(f"{cfg.resume_from}/optim_d.state_dict.pth"))

    lossf_g, lossf_d = get_lossf(cfg.lossf)
    loss_d_ema = lossf_d.optimal_val
    base_d_lr = cfg.optim_d['params']['lr']
    d_lr_multiplier = 1.
    use_wgan_gradient_penalty = cfg.use_gp
    use_r1_gradient_penalty = cfg.use_r1_gp
    use_r2_gradient_penalty = cfg.use_r2_gp
    loss_gp = torch.tensor(0.)
    loss_r1_gp = torch.tensor(0.)
    loss_r2_gp = torch.tensor(0.)
    adaptive_lr = cfg.mind_the_gap
    current_d_lr_factor = 1.
    skip_confident_model = cfg.skip_confident_model
    skip_d = False
    skip_g = False
    g_steps_trained = 0.
    d_steps_trained = 0.

    fixed_generator_noise = torch.randn([4, latent_size], device=cfg.device)

    epochs = cfg.num_iters * batch_size / len(dataset)
    print(f'Starting trainig for {epochs:.2f} epochs')

    net_g.train()
    net_d.train()
    pbar = tqdm(range(cfg.num_iters), desc='training...')
    resume_step = -1
    for i in range(cfg.num_iters):
        if i < resume_step:
            continue
        # get data
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)
        batch = batch.to(cfg.device)

        # update schedules
        cur_base_g_lr = cosine_decay_with_warmup(i, **cfg.lr_scheduler)
        cur_beta2 = cosine_decay_with_warmup(i, **cfg.beta2_scheduler)
        for group in optim_g.param_groups:
            group['lr'] = cur_base_g_lr
            group['beta2'] = cur_beta2
        base_d_lr = cosine_decay_with_warmup(i, **cfg.lr_scheduler)
        if not adaptive_lr:
            for group in optim_d.param_groups:
                group['lr'] = base_d_lr
                group['beta2'] = cur_beta2
        if use_r1_gradient_penalty or use_r2_gradient_penalty:
            gp_gamma = cosine_decay_with_warmup(i, **cfg.gamma_scheduler)

        # update d
        latent_noise = torch.randn([batch.shape[0], latent_size], device=cfg.device)
        optim_d.zero_grad()
        optim_g.zero_grad()
        with torch.no_grad():
            fake_samples = net_g(latent_noise).requires_grad_(True)
        batch = batch.detach().requires_grad_(True)
        pred_real = net_d(batch)
        pred_fake = net_d(fake_samples)
        if use_r1_gradient_penalty:
            loss_r1_gp = zero_centered_gp_lossf(batch, pred_real, gp_gamma)
        if use_r2_gradient_penalty:
            loss_r2_gp = zero_centered_gp_lossf(fake_samples, pred_fake, gp_gamma)
        loss_d = lossf_d(pred_real, pred_fake)
        if adaptive_lr:
            # update loss ema
            loss_d_ema = 0.95 * loss_d_ema + 0.05 * loss_d.item()
            # get new lr
            delta_v = loss_d_ema - lossf_d.optimal_val
            if delta_v < 0:
                d_lr_multiplier = max(lossf_d.h_min, lossf_d.h_min**(-delta_v/lossf_d.x_min))
            elif delta_v > 0:
                d_lr_multiplier = min(lossf_d.f_max, lossf_d.f_max**(-delta_v/lossf_d.x_max))
            else:
                d_lr_multiplier = 1
            # change d lr
            for group in optim_d.param_groups:
                group['lr'] = base_d_lr * d_lr_multiplier
                group['beta2'] = cur_beta2
        if skip_confident_model:
            delta_v = loss_d.item() - lossf_d.optimal_val
            if delta_v < lossf_d.confidence_min: # skip d update if it's too sure 
                skip_d = True
            elif delta_v > lossf_d.confidence_max: # skip g update if d is too bad
                skip_g = True
                
        total_d_loss = loss_d + loss_r1_gp + loss_r2_gp
        if use_r1_gradient_penalty:
            total_d_loss += loss_r1_gp
        if use_r2_gradient_penalty:
            total_d_loss += loss_r2_gp
        if use_wgan_gradient_penalty:
            loss_gp = gp_lossf(net_d, batch, fake_samples)
            total_d_loss += loss_gp
        if not skip_d:
            total_d_loss.backward()
            optim_d.step()
            d_steps_trained += 1
        d_pred_fake_item = pred_fake.mean().item()
        d_pred_real_item = pred_real.mean().item()

        # update g
        if not skip_g:
            latent_noise = torch.randn([batch.shape[0], latent_size], device=cfg.device)
            optim_d.zero_grad()
            optim_g.zero_grad()
            fake_samples = net_g(latent_noise)
            pred_fake = net_d(fake_samples)
            loss_g = lossf_g(pred_fake, disc=net_d, real_samples=batch)
            loss_g.backward()
            optim_g.step()
            g_steps_trained += 1
        else:
            loss_g = torch.tensor(torch.nan)

        # metrics
        descr = f"Step: {i+1:5}, G loss {loss_g.item():.4f}, D loss {loss_d.item():.4f}, D real {d_pred_real_item:.4f}, D fake {d_pred_fake_item:.4f}"
        if use_wgan_gradient_penalty:
            descr = descr + f", GP {loss_gp.item():.4f}"
        if use_r1_gradient_penalty:
            descr = descr + f", R1 {loss_r1_gp.item():.4f}"
        if use_r2_gradient_penalty:
            descr = descr + f", R2 {loss_r2_gp.item():.4f}"
        if adaptive_lr:
            descr = descr + f", D loss EMA {loss_d_ema.item():.4f}"
        if skip_confident_model:
            descr = descr + f', G trained: {not skip_g}, D trained: {not skip_d}'
            skip_g = False
            skip_d = False
        pbar.set_description(descr)
        if (i + 1) % cfg.log_every == 0:
            logger.log_metrics(
                {
                    'loss_g': loss_g.item(),
                    'loss_d': loss_d.item(),
                    'd_pred_real': d_pred_real_item,
                    'd_pred_fake': d_pred_fake_item,
                    'lr': cur_base_g_lr,
                }, i+1
            )
            if use_wgan_gradient_penalty:
                logger.log_metric(loss_gp.item(), 'loss_gp', i+1)
            if use_r1_gradient_penalty:
                logger.log_metric(loss_r1_gp.item(), 'loss_r1_gp', i+1)
            if use_r2_gradient_penalty:
                logger.log_metric(loss_r2_gp.item(), 'loss_r2_gp', i+1)
            if adaptive_lr:
                logger.log_metric(current_d_lr_factor, 'd_lr_factor', i+1)
                logger.log_metric(loss_d_ema, 'loss_d_ema', i+1)
            if skip_confident_model:
                logger.log_metric(d_steps_trained / cfg.log_every, 'd_steps_trained', i+1)
                logger.log_metric(g_steps_trained / cfg.log_every, 'g_steps_trained', i+1)

        if (i + 1) % cfg.img_every == 0:
            net_g.eval()
            with torch.no_grad():
                fixed_images = net_g(fixed_generator_noise).cpu()
            fixed_images = make_grid(fixed_images,
                                     normalize=True,
                                     value_range=(-1, 1))
            fixed_images = fixed_images.permute(1, 2, 0).numpy() * 255.
            fixed_images_pil = Image.fromarray(np.uint8(fixed_images))
            logger.log_image(fixed_images_pil, i+1)
            net_g.train()

        if (i + 1) % cfg.save_every == 0:
            # TODO move this logic to logger class
            os.makedirs(f'experiments/{experiment_str}/states/{i+1:05}')
            torch.save(net_g.state_dict(), f'experiments/{experiment_str}/states/{i+1:05}/net_g.state_dict.pth')
            torch.save(net_d.state_dict(), f'experiments/{experiment_str}/states/{i+1:05}/net_d.state_dict.pth')
            torch.save(optim_g.state_dict(), f'experiments/{experiment_str}/states/{i+1:05}/optim_g.state_dict.pth')
            torch.save(optim_d.state_dict(), f'experiments/{experiment_str}/states/{i+1:05}/optim_d.state_dict.pth')

        pbar.update(n=1)
    logger.finish()
    torch.save(net_g.state_dict(), f'experiments/{experiment_str}/net_g.state_dict.pth')
    torch.save(net_d.state_dict(), f'experiments/{experiment_str}/net_d.state_dict.pth')
    print('training finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='configs/train.yml')
    args = parser.parse_args()
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    cfg = argparse.Namespace(**cfg)

    train(cfg)
