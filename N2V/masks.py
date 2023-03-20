import torch
import torch.nn.functional as F
import numpy as np


class Mask:
    def __init__(self, n_masks, search_area, direction):
        self.n_masks = n_masks
        self.search_area = search_area
        self.direction = direction

    def forward(self, x):
        if self.direction == 'column':
            x = x.rot90(1, dims=[-2, -1])

        pad = (self.search_area[0] // 2, self.search_area[1] // 2)

        x_medians = F.pad(x, (pad[0], pad[0], pad[1], pad[1]), mode='constant', value=float('nan'))
        x_medians = x_medians.unfold(2, self.search_area[0], 1).unfold(3, self.search_area[1], 1).contiguous()

        x_medians[...,  self.search_area[0]//2, :] = float('nan')

        x_medians = x_medians.flatten(start_dim=-2, end_dim=-1)
        x_medians = x_medians.nanquantile(0.5, dim=-1)[0]

        rows = np.random.choice(range(x.shape[-2]), size=self.n_masks, replace=False)
        columns = np.arange(x.shape[-1])

        mask = torch.zeros_like(x_medians)
        mask[..., rows, :] = 1

        x_medians = x_medians * mask

        mask_invert = torch.ones_like(x)
        mask_invert[..., rows, :] = 0

        x = x * mask_invert
        x = x + x_medians

        if self.direction == 'column':
            x = x.rot90(3, dims=[-2, -1])
            rows_ = columns
            columns_ = rows
            rows = rows_
            columns = columns_

        return x, rows, columns


class RowMask(Mask):
    def __init__(self, n_masks, search_area):
        super().__init__(n_masks, search_area, direction='row')


class ColumnMask(Mask):
    def __init__(self, n_masks, search_area):
        super().__init__(n_masks, search_area, direction='column')


class CrossMask:
    def __init__(self, n_masks, search_area):
        self.n_masks = n_masks
        self.search_area = search_area

    def forward(self, x):
        pad = (self.search_area[0] // 2, self.search_area[1] // 2)

        rows = np.random.choice(range(x.shape[-2]), size=self.n_masks, replace=False)
        columns = np.random.choice(range(x.shape[-1]), size=self.n_masks, replace=False)

        masked = []
        for i in range(self.n_masks):
            x_ = x.clone()
            x_[..., rows[i], :] = float('nan')
            x_[..., :, columns[i]] = float('nan')
            x_ = F.pad(x_, (pad[0], pad[0], pad[1], pad[1]), mode='constant', value=float('nan'))
            x_ = x_.unfold(2, self.search_area[0], 1).unfold(3, self.search_area[1], 1)
            x_ = x_.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.search_area[0] * self.search_area[1])
            x_ = torch.nanquantile(x_, 0.5, dim=-1)

            mask = torch.zeros_like(x)
            mask[..., rows[i], :] = 1
            mask[..., :, columns[i]] = 1
            x_ = x_ * mask
            masked.append(x_)

        masked = torch.max(torch.stack(masked), dim=0)[0]

        mask_whole = torch.ones_like(x)
        mask_whole[..., rows, :] = 0
        mask_whole[..., :, columns] = 0

        x = x * mask_whole
        x = x + masked
        return x, rows, columns
      
