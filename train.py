import argparse
import yaml
import torch
import numpy as np
from data import get_dataset
from models import get_model
from losses import get_lossf, gp_lossf
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import os
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd

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
    experiment_str = f"{cfg.data['type']}-{cfg.net_d['type']}-{cfg.lossf}"
    if cfg.net_d['params']['use_spectral_norm']:
        experiment_str = experiment_str + "-sn"
    if cfg.use_gp:
        experiment_str = experiment_str + "-gp"
    if cfg.mind_the_gap:
        experiment_str = experiment_str + "-gap"
    if cfg.skip_confident_model:
        experiment_str = experiment_str + "-skip"
    os.makedirs(f'experiments/{experiment_str}/images', exist_ok=True)

    dataset = get_dataset(cfg.data)
    dataloader = DataLoader(dataset, shuffle=True, **cfg.data['loader'])
    train_iter = iter(dataloader)
    batch_size = cfg.data['loader']['batch_size']
    latent_size = 100

    net_g = get_model(cfg.net_g).to(cfg.device)
    net_d = get_model(cfg.net_d).to(cfg.device)

    optim_g = get_optimizer(net_g, cfg.optim_g)
    optim_d = get_optimizer(net_d, cfg.optim_d)

    lossf_g, lossf_d = get_lossf(cfg.lossf)
    loss_d_ema = lossf_d.optimal_val * 0.5 
    use_gradient_penalty = cfg.use_gp
    loss_gp = torch.tensor(0.)
    adaptive_lr = cfg.mind_the_gap
    current_d_lr_factor = 1.
    skip_confident_model = cfg.skip_confident_model
    g_steps_trained = 0.
    d_steps_trained = 0.

    fixed_generator_noise = torch.randn([4, 100], device=cfg.device)

    history = defaultdict(lambda: {})

    net_g.train()
    net_d.train()
    pbar = tqdm(range(cfg.num_iters), desc='training...')
    for i in pbar:
        # get data
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)
        batch = batch.to(cfg.device)

        # update d
        latent_noise = torch.randn([batch_size, latent_size], device=cfg.device)
        optim_d.zero_grad()
        optim_g.zero_grad()
        with torch.no_grad():
            fake_samples = net_g(latent_noise)
        pred_real = net_d(batch)
        pred_fake = net_d(fake_samples)
        loss_d = lossf_d(pred_real, pred_fake)
        total_d_loss = loss_d
        if use_gradient_penalty:
            loss_gp = gp_lossf(net_d, batch, fake_samples)
            total_d_loss += loss_gp
        total_d_loss.backward()
        optim_d.step()
        d_steps_trained += 1

        # update g
        latent_noise = torch.randn([batch_size, latent_size], device=cfg.device)
        optim_d.zero_grad()
        optim_g.zero_grad()
        fake_samples = net_g(latent_noise)
        pred_fake = net_d(fake_samples)
        loss_g = lossf_g(pred_fake, disc=net_d, real_samples=batch)
        loss_g.backward()
        optim_g.step()
        g_steps_trained += 1

        # metrics
        pbar.set_description(f"Step: {i+1:5}, G loss {loss_g.item():.4f}, D loss {loss_d.item():.4f}, GP {loss_gp.item():.4f}")
        if (i + 1) % cfg.log_every == 0:
            history['loss_g'][i+1] = loss_g.item()
            history['loss_d'][i+1] = loss_d.item()
            if use_gradient_penalty:
                history['loss_gp'][i+1] = loss_gp.item()
            if adaptive_lr:
                history['d_lr_factor'][i+1] = current_d_lr_factor
                history['loss_d_ema'][i+1] = loss_d_ema
            if skip_confident_model:
                history['d_steps_trained'][i+1] = d_steps_trained / cfg.log_every
                history['g_steps_trained'][i+1] = g_steps_trained / cfg.log_every

        if (i + 1) % cfg.fid_every == 0:
            net_g.eval()
            with torch.no_grad():
                fixed_images = net_g(fixed_generator_noise).cpu()
            fixed_images = make_grid(fixed_images,
                                     normalize=True,
                                     value_range=(-1, 1))
            fixed_images = fixed_images.permute(1, 2, 0).numpy() * 255.
            fixed_images_pil = Image.fromarray(np.uint8(fixed_images))
            fixed_images_pil.save(f'experiments/{experiment_str}/images/{i+1:5}.png')
            net_g.train()

        pbar.update(n=1)
    history_df = pd.DataFrame.from_dict(history)
    history_df.to_csv(f'experiments/{experiment_str}/history.csv')
    print('training finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default='configs/train.yml')
    args = parser.parse_args()
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    cfg = argparse.Namespace(**cfg)

    train(cfg)
