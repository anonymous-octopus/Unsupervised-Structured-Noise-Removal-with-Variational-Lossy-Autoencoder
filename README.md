# Unsupervised Structured Noise Removal with Variational Lossy Autoencoder

The scripts directory contains iPython notebooks with code for reproducing the experiments carried out in the paper.

scripts/CARE/: Our implementation of Content Aware Image Restoration https://doi.org/10.1038/s41592-018-0216-7

scripts/DVLAE/: Our proposed method.

scripts/HDN36/: Our implementation of HDN36 https://doi.org/10.48550/arXiv.2104.01374

scripts/N2V/: Our implementation of Structured N2V http://doi.org/10.1109/ISBI45749.2020.9098336

Dependencies.

We recommend installing packages to a conda environment with the following commands

`conda create -n DVLAE`

`conda activate DVLAE`

`conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

`conda install pytorch-lightning==1.9.3 -c conda-forge`

`conda install scikit-image`

`pip install matplotlib`

`pip install jupyterlab`

`pip install tensorboard`
