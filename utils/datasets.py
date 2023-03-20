import torch
import numpy as np


class TrainDatasetUnsupervised(torch.utils.data.Dataset):
    """PyTorch dataset for training dataset of images.

    Optionally iterates over the dataset more than once to account
    for random crops.

    Attributes:
        images: torch.tensor of noisy images.
        n_iters: int of number of times to iterate over dataset in one epoch.
        transform: torchvision.transforms of optional transformation to apply
        to images.
    """
    def __init__(self, images, n_iters=1, transform=None):
        self.images = images
        self.n_images = len(images)
        self.n_iters = n_iters
        self.transform = transform

    def __len__(self):
        return self.n_images * self.n_iters

    def __getitem__(self, idx):
        idx = idx % self.n_images
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image


class TrainDatasetSupervised(torch.utils.data.Dataset):
    """PyTorch dataset for training dataset of images.

    Optionally iterates over the dataset more than once to account
    for random crops.

    Attributes:
        low_snr: torch.tensor of noisy images.
        high_snr: torch.tensor of clean images.
        n_iters: int of number of times to iterate over dataset in one epoch.
        transform: torchvision.transforms of optional transformation to apply
        to images.
    """
    def __init__(self, low_snr, high_snr, n_iters=1, transform=None):
        self.low_snr = low_snr
        self.high_snr = high_snr
        self.n_images = len(low_snr)
        self.n_iters = n_iters
        self.transform = transform

    def __len__(self):
        return self.n_images * self.n_iters

    def __getitem__(self, idx):
        idx = idx % self.n_images
        low_snr = self.low_snr[idx]
        high_snr = self.high_snr[idx]
        if self.transform:
            seed = np.random.randint(1000)
            torch.manual_seed(seed)
            low_snr = self.transform(low_snr)
            torch.manual_seed(seed)
            high_snr = self.transform(high_snr)

        return low_snr, high_snr


class PredictDatasetUNet(torch.utils.data.Dataset):
    """PyTorch dataset for evaluation dataset of images.

    Attributes:
        images: torch.tensor of noisy images.
    """
    def __init__(self, images):
        self.n_images = len(images)
        self.images = images

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


class PredictDatasetVAE(torch.utils.data.Dataset):
    """PyTorch dataset for evaluation dataset of images.

    Replicates images 'n_samples' number of times so that the same image
    can be passed through the network multiple times to get multiple estimates
    of its clean signal in one go. The dataloader this dataset is passed to should
    then have a batch size equal to n_samples and should not shuffle.

    Attributes:
        images: torch.tensor of noisy images.
        n_samples: number of times a single image should be passed through the network.
    """
    def __init__(self, images, n_samples=10):
        self.n_samples = n_samples
        self.n_images = len(images) * n_samples
        if type(images) == torch.Tensor:
            self.images = images.repeat_interleave(n_samples, dim=0)
        elif type(images) == list:
            self.images = [[image] * n_samples for image in images]
            self.images = sum(self.images, [])

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image = self.images[idx]
        return image
      
