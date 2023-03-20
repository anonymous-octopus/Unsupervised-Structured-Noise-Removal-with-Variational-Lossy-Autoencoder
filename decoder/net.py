import torch
from torch import nn


class SDecoder(nn.Module):
    def __init__(self, colour_channels, code_features, n_filters=32, n_layers=4, kernel_size=3):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(code_features, n_filters, kernel_size, padding=kernel_size // 2, padding_mode='reflect'),
            nn.ReLU())

        self.hidden_convs = nn.ModuleList(
            [nn.Conv2d(n_filters, n_filters, kernel_size, padding=kernel_size // 2, padding_mode='reflect'),
             nn.ReLU()] * (n_layers - 2))

        self.out_conv = nn.Conv2d(n_filters * 2, colour_channels, kernel_size, padding=kernel_size // 2,
                                  padding_mode='reflect')

    def forward(self, s_code):
        s_code = self.in_conv(s_code)
        skip = s_code

        for layer in self.hidden_convs:
            s_code = layer(s_code)

        s_code = torch.cat((s_code, skip), dim=1)
        s = self.out_conv(s_code)

        return s

    def get_s(self, s_code, x):
        s = self.forward(s_code)
        mse = (s - x) ** 2
        return s, mse
