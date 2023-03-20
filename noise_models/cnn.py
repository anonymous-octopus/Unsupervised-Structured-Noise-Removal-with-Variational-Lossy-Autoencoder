import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical


class ShiftedConvolution(nn.Module):
    """Causal convolution for images.

    Pads image above and to the left, convolves with a masked kernel and returns
    an image of the same shape.

    Attributes:
        in_channels (int): Number of channels of input image.
        out_channels (int): Number of filters.
        kernel_mask (torch.Tensor): Array of ones and zeroes that will be multiplied with kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_mask):
        super().__init__()

        top_shift = kernel_mask.shape[0] - 1
        left_shift = kernel_mask.shape[1] - 1

        self.pad = nn.ConstantPad2d((left_shift, 0, top_shift, 0), 0)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              (kernel_mask.shape[0], kernel_mask.shape[1]))

        self.register_buffer('kernel_mask', kernel_mask[None, None])

    def forward(self, x):
        x = self.pad(x)
        self.conv.weight.data *= self.kernel_mask
        x = self.conv(x)
        return x


class ShiftedBlock(nn.Module):
    """Applies causal convolution, activation function and residual connection.

    Attributes:
        in_channels (int): Number of channels of input image.
        out_channels (int): Number of filters.
        code_features (int): Number of channels of the latent code that came from the main VAE.
        kernel_mask (torch.Tensor): Array of ones and zeros that will be multiplied with causal convolution kernel.
    """
    def __init__(self, in_channels, out_channels, code_features, kernel_mask):
        super().__init__()

        self.in_conv = ShiftedConvolution(in_channels, out_channels, kernel_mask)
        self.s_conv = nn.Conv2d(code_features, out_channels, 1)
        self.act_fn = nn.ReLU()
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1)

        if out_channels == in_channels:
            self.do_skip = True
        else:
            self.do_skip = False

    def forward(self, x, s_code):
        feat = self.in_conv(x) + self.s_conv(s_code)
        feat = self.act_fn(feat)
        out = self.out_conv(feat)

        if self.do_skip:
            out = out + x

        return out


class Block(nn.Module):
    """Applies 1x1 convolution, activation function and residual connection.

    Attributes:
        in_channels (int): Number of channels of input image.
        out_channels (int): Number of filters.
        code_features (int): Number of channels of the latent code that came from the main VAE.
    """
    def __init__(self, in_channels, out_channels, code_features):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.s_conv = nn.Conv2d(code_features, out_channels, 1)
        self.act_fn = nn.ReLU()
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1)

        if out_channels == in_channels:
            self.do_skip = True
        else:
            self.do_skip = False

    def forward(self, x, s_code):
        feat = self.in_conv(x) + self.s_conv(s_code)
        feat = self.act_fn(feat)
        out = self.out_conv(feat)

        if self.do_skip:
            out = out + x

        return out


def sample_mixture_model(weights, loc, scale):
    """Takes Gaussian mixture model parameters and returns a random sample from that distribution.

    Args:
        weights, loc, scale (torch.Tensor): Parameters of Gaussian mixture model.

    Returns:
        torch.Tensor: Random sample from GMM with same shape as parameters.
    """
    mixture_samples = Normal(loc, scale).rsample()
    component_idx = Categorical(weights.moveaxis(1, 3)).sample()

    return torch.gather(mixture_samples, dim=1, index=component_idx.unsqueeze(dim=1))


class PixelCNN(nn.Module):
    """CNN with causal convolutions.

    Applies masked causal convolution on first layer, all subsequent layers are 1x1 convolutions.

    Attributes:
        colour_channels (int): Number of colour channels of input image.
        code_features (int): Number of channels of latent code from main VAE.
        kernel_mask (torch.Tensor): Array of ones and zeros that will be multiplied with kernel of causal convolution.
        n_filters (int): Number of filters.
        n_layers (int): Number of layers.
        n_gaussians (int): Number of components in Gaussian mixture model.
    """
    def __init__(self,
                 colour_channels,
                 code_features,
                 kernel_mask,
                 n_filters=16,
                 n_layers=4,
                 n_gaussians=5):
        super().__init__()
        self.n_gaussians = n_gaussians

        assert n_layers >= 2

        kernel_mask[-1, -1] = 0

        out_channels = colour_channels * n_gaussians * 3

        self.shifted_convs = nn.ModuleList(
            [ShiftedBlock(colour_channels,
                          n_filters,
                          code_features,
                          kernel_mask)] + \
            [Block(n_filters,
                   n_filters,
                   code_features)] * (n_layers - 2) + \
            [Block(n_filters,
                   out_channels,
                   code_features)]
        )

    def forward(self, x, s_code):
        for layer in self.shifted_convs:
            x = layer(x, s_code)
        return x

    def extract_params(self, params):
        """Turns output of network into separate Gaussian mixture model parameters.

        Args:
            params (torch.Tensor): Output of network.
        """
        # Make weights positive and sum to 1
        weights = params[:, 0::3].unfold(1, self.n_gaussians, self.n_gaussians)
        weights = nn.functional.softmax(weights, dim=-1)

        loc = params[:, 1::3].unfold(1, self.n_gaussians, self.n_gaussians)

        # Make scales positive
        scale = params[:, 2::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = scale.exp()
        return weights, loc, scale

    def loglikelihood(self, x, s_code):
        """Calculates reconstruction loglikelihood.

        Args:
            x (torch.Tensor): Image to be reconstructed.
            s_code (torch.Tensor): Latent code of image being reconstructed

        Returns:
            torch.Tensor: Loglikelihood of x given s_code.
        """
        params = self.forward(x, s_code)

        weights, loc, scale = self.extract_params(params)  # returns [B, C, H, W, G]

        loglikelihoods = Normal(loc, scale).log_prob(x[..., None])
        temp = loglikelihoods.max(dim=-1, keepdim=True)[0]
        loglikelihoods = loglikelihoods - temp
        loglikelihoods = loglikelihoods.exp()
        loglikelihoods = loglikelihoods * weights
        loglikelihoods = loglikelihoods.sum(dim=-1, keepdim=True)
        loglikelihoods = loglikelihoods.log()
        loglikelihoods = loglikelihoods + temp

        return loglikelihoods
      
