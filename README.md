# Know Your GAN
A playground for testing methods of training Generative Adversarial Networks.

---

## Roadmap
- [ ] Img2Img extension
- [ ] Post experiment results
- [ ] Requirements
- [ ] Do some cleaning
- [x] Release the repo

## Requirements
To be written soon.

## Getting started
Firstly, you should download data and adjust [`.yml file`](./configs/train.yml), then run
```
python train.py --config ./configs/train.yml
```
## Methods

Original GAN ([Paper](https://arxiv.org/abs/1406.2661)):
- Saturating GAN loss
- Non-saturating GAN loss

Wasserstein GAN
- Wasserstein GAN loss ([Paper](https://arxiv.org/abs/1701.07875))
- Gradient penalty (WGAN-GP) ([Paper](https://arxiv.org/abs/1704.00028))

Least Squares GAN ([Paper](https://arxiv.org/abs/1611.04076)):
- LSGAN loss

Spectral Normalization ([Paper](https://arxiv.org/abs/1802.05957)):
- Spectral normalization for discriminator

Geometric GAN ([Paper](https://arxiv.org/abs/1705.02894)):
- Hinge loss

Relativistic Discriminator ([Paper](https://arxiv.org/abs/1807.00734)):
- Relativistic GAN losses
- Relativistic Average GAN losses

Mind the (optimality) Gap ([Paper](https://arxiv.org/abs/2302.00089)): 
- Gap-Aware LR scheduler

Image-to-Image transition ([Paper](https://arxiv.org/pdf/1611.07004)):
- PatchGAN architecture

U-Net based discriminator ([Paper](https://arxiv.org/abs/2002.12655)):
- Idea of U-Net architecture for discriminator

Mostly inspired from [R3GAN](https://github.com/brownvc/R3GAN) ([Paper](https://arxiv.org/abs/2501.05441)):
- Generator and Discriminator architectures.
- Zero-centered gradient penalties
- RpGAN loss

Thanks [ChristophReich1996](https://github.com/ChristophReich1996) for an awesome [repo](https://github.com/ChristophReich1996/Mode_Collapse) that has inspired this toybox.

## Metrics

Following metrics are supported:


* `fid50k_full`: Fr&eacute;chet inception distance against the full dataset.
* `kid50k_full`: Kernel inception distance against the full dataset.
* `pr50k3_full`: Precision and recall against the full dataset.

Refer to the StyleGAN3 code base for more details.
