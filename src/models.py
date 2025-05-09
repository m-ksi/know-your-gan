import torch
import torch.nn as nn
import math
from torch.nn.utils import spectral_norm


def initialize(layer, act_gain=1):
    fan_in = layer.weight.data.size(1) * layer.weight.data[0][0].numel()
    if act_gain > 0:
        layer.weight.data.normal_(0, act_gain / math.sqrt(fan_in))
    else:
        layer.weight.data.fill_(0.0)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def get_model(conf):
    match conf["type"]:
        case "ImageGenerator":
            cl = ImageGenerator
        case "PatchDiscriminator":
            cl = PatchDiscriminator
        case "UNetDiscriminator":
            cl = UNetDiscriminator
        case "SymmetricDiscriminator":
            cl = SymmetricDiscriminator
        case _:
            raise NotImplementedError(f"Model type {conf['type']} not implemented!")
    model = cl(**conf["params"])
    return model


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        cardinality,
        expansion_f,
        ks,
        var_scaling_param,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        expanded_ch = in_ch * expansion_f
        act_gain = math.sqrt(2 / (1 + 0.2**2)) * var_scaling_param ** (-1 / (2 * 3 - 2))

        self.conv1 = initialize(
            nn.Conv2d(
                in_ch, expanded_ch, kernel_size=1, stride=1, padding=0, bias=True
            ),
            act_gain,
        )
        self.conv2 = initialize(
            nn.Conv2d(
                expanded_ch,
                expanded_ch,
                kernel_size=ks,
                stride=1,
                padding=1,
                groups=cardinality,
                bias=True,
            ),
            act_gain,
        )
        self.conv3 = initialize(
            nn.Conv2d(
                expanded_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=False
            ),
            0,
        )
        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            # self.conv3 = spectral_norm(self.conv3)

    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        h = self.conv3(h)
        return x + h


class GenerativeBasis(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(out_ch, 4, 4).normal_(0, 1))
        self.fc = initialize(nn.Linear(in_ch, out_ch, bias=False))

    def forward(self, x):
        return self.basis.view(1, -1, 4, 4) * self.fc(x).view(x.shape[0], -1, 1, 1)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        h = self.conv(x)
        h = nn.functional.interpolate(
            h, scale_factor=2, mode="bilinear", align_corners=False
        )
        return h


class DiscriminatorHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.basis = initialize(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=4,
                stride=1,
                padding=0,
                groups=in_ch,
                bias=False,
            )
        )
        self.fc = initialize(nn.Linear(in_ch, out_ch, bias=False))

    def forward(self, x):
        return self.fc(self.basis(x).view(x.shape[0], -1))


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, use_spectral_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        h = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        h = self.conv(h)
        return h


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        cardinality,
        num_blocks,
        expansion_f,
        ks,
        var_scaling_param,
        noise_inp=True,
    ):
        super().__init__()
        self.noise_inp = noise_inp
        if noise_inp:
            in_layer = GenerativeBasis(in_ch, out_ch)
        else:
            in_layer = Upsample(in_ch, out_ch)

        self.layers = nn.Sequential(
            in_layer,
            *[
                ResBlock(out_ch, cardinality, expansion_f, ks, var_scaling_param)
                for i in range(num_blocks)
            ],
        )

    def forward(self, x):
        h = self.layers(x)
        return h


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        cardinality,
        num_blocks,
        expansion_f,
        ks,
        var_scaling_param,
        head=False,
        use_spectral_norm=False,
    ):
        super().__init__()
        top = (
            DiscriminatorHead(in_ch, out_ch)
            if head
            else Downsample(in_ch, out_ch, use_spectral_norm=use_spectral_norm)
        )
        self.layers = nn.Sequential(
            *[
                ResBlock(
                    in_ch,
                    cardinality,
                    expansion_f,
                    ks,
                    var_scaling_param,
                    use_spectral_norm=use_spectral_norm,
                )
                for _ in range(num_blocks)
            ],
            top,
        )

    def forward(self, x):
        return self.layers(x)


class ImageGenerator(nn.Module):
    def __init__(self, nz, channels, cardinalities, n_blocks, expansion_f, ks=3):
        super().__init__()
        self.z_dim = nz
        var_scaling_param = sum(n_blocks)
        layers = [
            GeneratorBlock(
                nz,
                channels[0],
                cardinalities[0],
                n_blocks[0],
                expansion_f,
                ks,
                var_scaling_param,
            )
        ]
        layers += [
            GeneratorBlock(
                channels[i],
                channels[i + 1],
                cardinalities[i + 1],
                n_blocks[i + 1],
                expansion_f,
                ks,
                var_scaling_param,
                noise_inp=False,
            )
            for i in range(len(channels) - 1)
        ]
        layers.append(initialize(nn.Conv2d(channels[-1], 3, kernel_size=1)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SymmetricDiscriminator(nn.Module):
    def __init__(
        self,
        channels,
        cardinalities,
        n_blocks,
        expansion_f,
        ks=3,
        use_spectral_norm=False,
    ):
        super().__init__()
        var_scaling_param = sum(n_blocks)
        self.extraction_layer = initialize(nn.Conv2d(3, channels[0], kernel_size=1))
        self.layers = nn.Sequential(
            *[
                DiscriminatorBlock(
                    channels[x],
                    channels[x + 1],
                    cardinalities[x],
                    n_blocks[x],
                    expansion_f,
                    ks,
                    var_scaling_param,
                    use_spectral_norm=use_spectral_norm,
                )
                for x in range(len(channels) - 1)
            ],
            DiscriminatorBlock(
                channels[-1],
                1,
                cardinalities[-1],
                n_blocks[-1],
                expansion_f,
                ks,
                var_scaling_param,
                head=True,
                use_spectral_norm=use_spectral_norm,
            ),
        )

    def forward(self, x):
        h = self.extraction_layer(x)
        h = self.layers(h)
        return h.view(x.shape[0])


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
            norm(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            norm(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, 1, 3, 1, 1),
        )
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0.0, 0.2)
            elif "bias" in name:
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
        self.ups = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv0 = nn.Conv2d(3, nf, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False))
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


if __name__ == "__main__":
    import torchsummary
    # g = SymmetricDiscriminator(
    #     [128, 128, 128, 192],
    #     [16, 16, 16, 24],
    #     [2, 2, 2, 2],
    #     2,
    #     use_spectral_norm=True
    # ).cuda()
    g = SymmetricDiscriminator(
        [192, 192, 192, 256],
        [24, 24, 24, 32],
        [2, 2, 2, 2],
        2,
        use_spectral_norm=False
    ).cuda()
    # g = UNetDiscriminator(32, True).cuda()
    # g = PatchDiscriminator(64).cuda()
    torchsummary.summary(g, (3, 32, 32), batch_size=2)
    # print(g(torch.randn(2, 3, 32, 32).cuda()))

    #   type: SymmetricDiscriminator
    # params:
    # channels: [128, 128, 128, 192]
    # cardinalities: [16, 16, 16, 24]
    # n_blocks: [2, 2, 2, 2, 2]
    # expansion_f: 2
    # use_spectral_norm: false

    # g = ImageGenerator(
    #     **{
    #         "nz": 64,
    #         "channels": [256, 192, 192, 192],
    #         "cardinalities": [32, 24, 24, 24],
    #         "n_blocks": [2, 2, 2, 2, 2],
    #         "expansion_f": 2,
    #     }
    # ).cuda()
    # torchsummary.summary(g, (64,), batch_size=2)
