import numpy as np
import torch
from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from ..lib.utils import (crop_img_tensor, pad_img_tensor)
from .lvae_layers import (TopDownLayer, BottomUpLayer, TopDownDeterministicResBlock, BottomUpDeterministicResBlock)


class N2V(LightningModule):
    """Hierarchical variational autoencoder.

    Encodes a noisy image, x, as a hierarchy of latent variables, z, and begins decoding by deterministically mapping
    z to a prediction of the underlying clean signal in x. The external noise model should then finish decoding z to
    x by predicting the noise that must be added to s, returning the loglikelihood of s, log[p(x|s)].

    Attributes:
        data_mean: A float of the mean of the noisy data. Used with data_std to normalise inputs.
        data_std: A float of the standard deviation of the noisy data.
        colour_channels: Int for number of colour channels in noisy images.
        img_shape: The spatial dimensions of the inputs as a list: [Height, Width] or tuple (Height, Width).
        noise_model: An object with a .loglikelihood() function.
        s_decoder: A separate network that learns to decode an s code into an s estimate
        z_dims: List of ints, one for the number of features of each latent variable in the hierarchy.
        blocks_per_layer: Int for the number of residual blocks at each level of the heirarchy.
        n_filters: Int for number of feature channels.
        res_block_type: String for the ordering of operations within each block. Options are: 'cabcab' or 'bacbac'.
        merge_type: String for how features from bottom-up pass will be merged with features from top-down pass.
        'residual' uses a residual block, 'linear' uses a single convolution.
        stochastic_skip: Bool for whether to use skip connections from previous layer of hierarchy.
        gated: Bool for whether to uses forget gate activation.
        batchnorm: Bool for use of batch normalisation.
        downsample: Bool for whether to downsample input.
        mode_pred: Bool. If false, losses will not be calculated.
    """

    def __init__(self,
                 data_mean,
                 data_std,
                 colour_channels,
                 img_shape,
                 mask_func,
                 s_decoder,
                 z_dims=None,
                 blocks_per_layer=5,
                 n_filters=64,
                 res_block_type='bacbac',
                 merge_type='residual',
                 stochastic_skip=True,
                 gated=True,
                 batchnorm=True,
                 downsample=True,
                 mode_pred=False):
        if z_dims is None:
            z_dims = [32] * 6
        self.save_hyperparameters(ignore=['noise_model', 's_decoder'])
        super().__init__()
        self.data_mean = data_mean
        self.data_std = data_std
        self.img_shape = tuple(img_shape)
        self.mask_func = mask_func
        self.do_mask = True  # Will alternate between true and false on training and validation steps
        self.s_decoder = s_decoder
        self.z_dims = z_dims
        self.n_layers = len(self.z_dims)
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.stochastic_skip = stochastic_skip
        self.gated = gated
        self.mode_pred = mode_pred

        # The s_decoder is trained alongside the main VAE, but uses its own optimizer
        self.automatic_optimization = False

        # Deterministic network takes the mode of the posterior distribution instead of sampling from it
        self.mode_layers = list(range(self.n_layers))

        # Number of downsampling steps per layer
        downsampling = [1] * self.n_layers if downsample else [0] * self.n_layers
        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(downsampling))

        assert max(downsampling) <= self.blocks_per_layer
        assert len(downsampling) == self.n_layers

        # First bottom-up layer: change num channels
        self.first_bottom_up = nn.Sequential(
            nn.Conv2d(colour_channels, n_filters, 5, padding=2, padding_mode='replicate'),
            nn.ELU(),
            BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                batchnorm=batchnorm,
                res_block_type=res_block_type,
            ))

        # Init lists of layers
        self.top_down_layers = nn.ModuleList([])
        self.bottom_up_layers = nn.ModuleList([])

        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                    gated=gated,
                ))

            # Add top-down stochastic layer at level i.
            # The architecture when doing inference is roughly as follows:
            #    p_params = output of top-down layer above
            #    bu = inferred bottom-up value at this layer
            #    q_params = merge(bu, p_params)
            #    z = stochastic_layer(q_params)
            #    possibly get skip connection from previous top-down layer
            #    top-down deterministic ResNet
            #
            # When doing generation only, the value bu is not available, the
            # merge layer is not used, and z is sampled directly from p_params.
            self.top_down_layers.append(
                TopDownLayer(
                    z_dim=z_dims[i],
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=downsampling[i],
                    merge_type=merge_type,
                    batchnorm=batchnorm,
                    stochastic_skip=stochastic_skip,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=res_block_type,
                    gated=gated,
                ))

        # Final top-down layer
        modules = list()
        for i in range(blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=n_filters,
                    c_out=n_filters,
                    batchnorm=batchnorm,
                    res_block_type=res_block_type,
                    gated=gated,
                ))
        self.final_top_down = nn.Sequential(*modules)

    def forward(self, x):
        img_size = x.size()[2:]

        # Pad x to have base 2 side lengths to make resampling steps simpler
        x_pad = self.pad_input(x)

        if self.do_mask:
            # Mask if not on a validation step
            x_pad, masked_rows, masked_columns = self.mask_func(x_pad)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)

        # Top-down inference/generation
        out, _ = self.topdown_pass(bu_values, mode_layers=self.mode_layers)

        # Restore original image size
        predicted_s_code = crop_img_tensor(out, img_size)

        # Decode latent code into an estimate of the signal with the s_decoder
        # Calculate the mean squared error between signal estimate and x
        predicted_s, mse = self.s_decoder.get_s(predicted_s_code, x)

        if self.do_mask is True:
            # If pixels have been masked, only calculate the loss on them
            mse = mse[..., masked_rows[:, None], masked_columns[None, :]].mean()
        else:
            mse = mse.mean()

        output = {
            'mse': mse,
            'predicted_s_code': predicted_s_code,
            'predicted_s': predicted_s,
        }
        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(self,
                     bu_values=None,
                     n_img_prior=None,
                     mode_layers=None,
                     constant_layers=None,
                     forced_latent=None):

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)

        # KL divergence of each layer
        kl = [None] * self.n_layers

        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # Top-down inference/generation loop
        out = None
        for i in reversed(range(self.n_layers)):
            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers

            # Input for skip connection
            skip_input = out  # TODO or out_pre_residual? or both?

            # Full top-down layer, including sampling and deterministic part
            out, kl_elementwise = self.top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred
            )
            kl[i] = kl_elementwise  # (batch, ch, h, w)

        # Final top-down layer
        out = self.final_top_down(out)

        return out, kl

    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x

    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = ("input size must be either (N, C, H, W) or (H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size

    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using
        dwnsc = self.overall_downscale_factor
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape

    def training_step(self, batch, _):
        # Normalise data
        x = (batch - self.data_mean) / self.data_std

        # Carry out masking on training steps
        self.do_mask = True
        # Returns dictionary containing predicted signal and loss terms
        model_out = self.forward(x)

        mse = model_out['mse']

        optimizer = self.optimizers()

        optimizer.zero_grad()
        self.manual_backward(mse)
        optimizer.step()

        self.log_dict({'mse': mse})

    def training_epoch_end(self, _):
        scheduler = self.lr_schedulers()
        scheduler.step(self.trainer.callback_metrics['val_mse'])

    def log_images_for_tensorboard(self, pred, x, img_mmse):
        clamped_input = torch.clamp((x - x.min()) / (x.max() - x.min()), 0, 1)
        clamped_pred = torch.clamp((pred - pred.min()) / (pred.max() - pred.min()), 0, 1)
        clamped_mmse = torch.clamp((img_mmse - img_mmse.min()) / (img_mmse.max() - img_mmse.min()), 0, 1)
        self.trainer.logger.experiment.add_image('inputs/img', clamped_input[0],
                                                 self.current_epoch)
        for i in range(2):
            self.trainer.logger.experiment.add_image('predcitions/sample_{}'.format(i), clamped_pred[i],
                                                     self.current_epoch)
        self.trainer.logger.experiment.add_image('predcitions/mmse (10 samples)', clamped_mmse[0],
                                                 self.current_epoch)

    def validation_step(self, batch, batch_idx):
        # Normalise data
        x = (batch - self.data_mean) / self.data_std

        self.do_mask = False
        model_out = self.forward(x)

        mse = model_out['mse']

        self.log_dict({'val_mse': mse})

        if batch_idx == 0:
            # Display validation set results on tensorboard
            # One noisy image, 10 predictions of its signal and their mean
            idx = np.random.randint(x.shape[0])
            all_samples = self.forward(torch.repeat_interleave(x[idx:idx + 1], 10, 0))['predicted_s']
            mmse = torch.mean(all_samples, 0, keepdim=True)
            self.log_images_for_tensorboard(all_samples, x[idx:idx + 1], mmse)

    def predict_step(self, batch, _):
        # Don't calculate loss or carry out masking
        self.mode_pred = True
        self.do_mask = False

        # x should be a batch of N identical images
        x = (batch - self.data_mean) / self.data_std
        # Returns N different predicted signals for the same noisy input
        out = self.forward(x)['predicted_s']
        out = out * self.data_std + self.data_mean
        return out

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         patience=50,
                                                         factor=0.5,
                                                         verbose=True)

        return [optimizer], [scheduler]
      
