import torch.nn as nn
from torch.nn.utils import spectral_norm

def get_model(conf):
    match conf['type']:
        case 'ImageGenerator':
            cl = ImageGenerator
        case 'PatchDiscriminator':
            cl = PatchDiscriminator
        case 'UNetDiscriminator':
            cl = UNetDiscriminator
        case _:
            raise NotImplementedError(f"Model type {conf['type']} not implemented!")
    model = cl(**conf['params'])
    return model

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(nf, nf*3, kernel_size=3, stride=1, padding=1, groups=nf, bias=True)
        self.conv3 = nn.Conv2d(nf*3, nf, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        h = self.conv3(h)
        return x + h
    
class GeneratorBlock(nn.Module):
    def __init__(self, nf_in, nf_out, noise_inp=False):
        super().__init__()
        self.noise_inp = noise_inp
        if noise_inp:
            self.in_layer = nn.Linear(nf_in, nf_out, bias=False)
        else:
            self.in_layer = nn.Conv2d(nf_in, nf_out, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.res1 = ResBlock(nf_out)
        self.res2 = ResBlock(nf_out)

    def forward(self, x):
        h = self.in_layer(x)
        if self.noise_inp:
            h = h.unsqueeze(-1).unsqueeze(-1)
        h = nn.functional.interpolate(h, scale_factor=2, mode='bilinear')
        h = self.res1(h)
        h = self.res2(h)
        return h

class ImageGenerator(nn.Module):
    def __init__(self, nz, ngf, out128=False):
        super().__init__()
        layers = [GeneratorBlock(nz, ngf*8, noise_inp=True)]
        layers.append(GeneratorBlock(ngf*8, ngf*4))
        if out128:
            layers.append(GeneratorBlock(ngf*4, ngf*4))
        layers.append(GeneratorBlock(ngf*4, ngf*2))
        if out128:
            layers.append(GeneratorBlock(ngf*2, ngf*2))
        layers.append(GeneratorBlock(ngf*2, ngf))
        layers.append(GeneratorBlock(ngf, ngf))
        layers.append(nn.Conv2d(ngf, 3, 3, 1, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, nf=32, use_spectral_norm=True):
        super().__init__()
        if use_spectral_norm:
            norm = spectral_norm
        else:
            def norm(x):
                return x
        self.conv = nn.Sequential(
            norm(nn.Conv2d(3, nf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            norm(nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            norm(nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, 1, 3, 1, 1),
        )
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, 0.0, 0.2)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        return self.conv(x)

class UNetDiscriminator(nn.Module):
    def __init__(self, nf=32, use_spectral_norm=True):
        super().__init__()
        if use_spectral_norm:
            norm = spectral_norm
        else:
            def norm(x):
                return x
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0 = nn.Conv2d(3, nf, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(nf, nf*2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(nf*2, nf*4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(nf*4, nf*8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(nf*8, nf*4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(nf*4, nf*2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(nf*2, nf, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)
    
    def forward(self, x):
        x0 = self.act(self.conv0(x))
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))

        x3 = self.ups(x3)
        x4 = self.act(self.conv4(x3))
        x4 = x4 + x2
        x4 = self.ups(x4)
        x5 = self.act(self.conv5(x4))
        x5 = x5 + x1
        x5 = self.ups(x5)
        x6 = self.act(self.conv6(x5))
        x6 = x6 + x0

        out = self.act(self.conv7(x6))
        out = self.act(self.conv8(out))
        out = self.conv9(out)
        return out
    
class SynthGenerator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent_size, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256, bias=True),
                                    nn.Tanh(),
                                    nn.Linear(256, 2, bias=True))
    
    def forward(self, x):
        return self.layers(x)

class SynthDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
            if use_spectral_norm:
                self.layers = nn.Sequential(spectral_norm(nn.Linear(2, 256, bias=True)),
                                    nn.LeakyReLU(),
                                    spectral_norm(nn.Linear(256, 256, bias=True)),
                                    nn.LeakyReLU(),
                                    spectral_norm(nn.Linear(256, 256, bias=True)),
                                    nn.LeakyReLU(),
                                    spectral_norm(nn.Linear(256, 256, bias=True)),
                                    nn.LeakyReLU(),
                                    spectral_norm(nn.Linear(256, 1, bias=True)))
            else:
                self.layers = nn.Sequential(nn.Linear(2, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 256, bias=True),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 1, bias=True))
                
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    import torchsummary
    g = ImageGenerator(100, 64)
    torchsummary.summary(g, (100,), batch_size=2)
