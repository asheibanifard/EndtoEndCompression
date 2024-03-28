import numpy as np
import torch

import nibabel as ni
from utils import LR_image_producer


def data_reader(path='../data/BRATS_001.nii.gz'):
    volume = ni.load(path).get_fdata().\
                transpose(2, 1, 0, 3)
    volume = (volume - volume.min())/(volume.max() - volume.min())
    vol = volume[10:140, 22:214, 22:214, 1]
    vol_patch = vol[:64, 64:128, 64:128]
    vol_patch = vol_patch.astype(np.float32)
    print(vol_patch.dtype)
    print('Volume shape:', vol_patch.shape)
    print('File size (Byte):', np.prod((vol_patch.shape))*4)
    print('Number_of_elements:', np.prod((vol_patch.shape)))
    print('bpp:', np.prod((vol_patch.shape))*4*8/np.prod((vol_patch.shape)))
    scale = 2
    hr = vol_patch
    np.save('hr.npy', hr)
    HR = torch.tensor(hr).unsqueeze(0).unsqueeze(0)
    LR = torch.tensor(LR_image_producer(hr, scale)).unsqueeze(0).unsqueeze(0)
    return HR, LR, vol_patch
