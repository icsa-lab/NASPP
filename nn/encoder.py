import torch
import torch.nn as nn

from .decoder import InvertedResidual, conv_bn_relu6

class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.inverted_residual_layer_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.out_sizes = [24, 32, 96, 320]
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.layer1 = conv_bn_relu6(3, input_channel, 2)
        self.n_layers = len(self.inverted_residual_layer_config)
        for idx, (t, c, n, s) in enumerate(self.inverted_residual_layer_config):
            output_channel = int(c * width_mult)
            features = []
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
            setattr(self, 'layer{}'.format(idx + 2), nn.Sequential(*features))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)
        l8 = self.layer8(l7)
        return l3, l4, l6, l8
