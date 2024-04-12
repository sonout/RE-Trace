import random
from typing import Tuple, List, Optional
from math import sqrt
import numpy as np

import torch
from torch import Tensor

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, traj):
        for t in self.transforms:
            traj = t(traj)
        return traj


class RandomFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, traj):
        if torch.rand(1) < self.p:
            return torch.flip(traj, [0])
        return traj


class RandomMasking(torch.nn.Module):
    def __init__(self, p=0.3, share = 0.5):
        super().__init__()
        self.p = p
        self.share = share

    def forward(self, traj):
        if torch.rand(1) < self.p:
            mask = torch.rand((traj.shape[0]), device=traj.device) > self.share # About share % will be True
            traj_masked = traj * mask.int()
            return traj_masked
        return traj


class RandomConsecutiveMasking(torch.nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, traj):
        #if torch.rand(1) < self.p:
        #    return F.hflip(img)
        raise NotImplementedError
        return traj

class RandomCropping(torch.nn.Module):
    def __init__(self, p=0.3, share_min=0.1, share_max=0.4):
        super().__init__()
        self.share_min = share_min
        self.share_max = share_max
        self.p = p

    def forward(self, traj):
        if torch.rand(1) < self.p:
            length = len(traj)
            # Pick share between share_min and share_max
            share = (self.share_min - self.share_max) * torch.rand(1) + self.share_max
            length_crop = int(length * share)

            # Decide whether to crop begining or end
            if torch.rand(1) < 0.5:
                traj_cropped = traj[length_crop:]
            else:
                traj_cropped = traj[:-length_crop]
            return traj_cropped
        return traj
    

##### Traj Augs from TrajCL
class Simplify(torch.nn.Module):
    def __init__(self, p=0.3, traj_simp_dist = 100):
        super().__init__()
        self.p = p
        self.traj_simp_dist = traj_simp_dist

    def forward(self, traj):
        # src: [[lon, lat], [lon, lat], ...]
        if torch.rand(1) < self.p:
            return rdp(traj, epsilon = self.traj_simp_dist)
        return traj


class Shift(torch.nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, traj):
        if torch.rand(1) < self.p:
            return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in traj]
        return traj


class Mask(torch.nn.Module):
    def __init__(self, p=0.3, traj_mask_ratio = 0.3):
        super().__init__()
        self.p = p
        self.traj_mask_ratio = traj_mask_ratio

    def forward(self, traj):
        if torch.rand(1) < self.p:
            l = len(traj)
            arr = np.array(traj)
            mask_idx = np.random.choice(l, int(l * self.traj_mask_ratio), replace = False)
            return np.delete(arr, mask_idx, 0).tolist()
        return traj


class Subset(torch.nn.Module):
    def __init__(self, p=0.3, traj_subset_ratio = 0.7):
        super().__init__()
        self.p = p
        self.traj_subset_ratio = traj_subset_ratio

    def forward(self, traj):
        if torch.rand(1) < self.p:
            l = len(traj)
            max_start_idx = l - int(l * self.traj_subset_ratio)
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + int(l * self.traj_subset_ratio)
            return traj[start_idx: end_idx]
        return traj



##### Helper functions

def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results

def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) -
            (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d
    
def truncated_rand(mu = 0, sigma = 0.5, factor = 100, bound_lo = -100, bound_hi = 100):
    # using the defaults parameters, the success rate of one-pass random number generation is ~96%
    # gauss visualization: https://www.desmos.com/calculator/jxzs8fz9qr?lang=zh-CN
    while True:
        n = random.gauss(mu, sigma) * factor
        if bound_lo <= n <= bound_hi:
            break
    return n