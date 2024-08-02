from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
import numpy as np
from monai.data import DataLoader, Dataset
from torch.cuda.amp import autocast
from torchvision.transforms import CenterCrop
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

from utils.utilization import get_config
from utils.transforms import (
    get_train_transform,
    get_test_transform,
    get_test_transform_no_scaling,
)
from utils.visualization import save_pred_img
from utils.const import (
    CODE_FOLDER,
    RESULT_FOLDER,
    DATA_FOLDER,
    MODEL_PARAMS_FOLDER,
    GAN_MODEL_PARAMS_FOLDER,
)
from utils.add_argument import add_argument
from utils.utils import seed_everything, get_train_val_test_subject_ids, get_data_path


def interpolation_img_by_dir(target_dir, ref_dir, ref_img):
    solution = torch.linalg.lstsq(ref_dir.T, target_dir.unsqueeze(1))
    coefs = solution.solution.squeeze()
    target_img = coefs[0] * ref_img[0]
    for i in range(1, 6):
        target_img += coefs[i] * ref_img[i]

    return target_img


ssim = StructuralSimilarityIndexMeasure()
SSIM_interp = []

test_subjects = [
    101309,106319,107018,108121,109830,112516,
    113215,120717,123117,123420,127630,129028,
    129331,130619,131419,131722,138231,144832,
    150726,151324,151829,159441,162228,169949,
    171330,171633,172534,361234,397760,413934]

test_subject_paths = get_data_path(
    test_subjects,
    Path("/path-to-data"),
    is_val=False,
    is_test=True,
    n_neighbor_directions=6,
)


test_transforms = get_test_transform()
test_transforms_no_scaling = get_test_transform_no_scaling()
test_ds = Dataset(data=test_subject_paths, transform=test_transforms_no_scaling)
test_loader = DataLoader(test_ds, batch_size=30, shuffle=False, num_workers=4)

n_test = len(test_loader)
roi_size = (145, 174)
transform_crop = CenterCrop(roi_size)

for step, data in enumerate(test_loader):

    # sequence:
    #cur_id 101309 direction 00 slice 000
    #cur_id 101309 direction 30 slice 000
    #cur_id 101309 direction 00 slice 001
    #cur_id 101309 direction 30 slice 001
    #cur_id 101309 direction 00 slice 002
    #cur_id 101309 direction 30 slice 002
    #cur_id 101309 direction 00 slice 003
    #cur_id 101309 direction 30 slice 003
    #...
    
    adjacent_directions = data["adjacent_directions"].transpose(-2, -1) # torch.Size([30, 3, 3])
    center_directions = data["center_directions"] # torch.Size([30, 3])
    input_imgs = data["adjacent_img"] # torch.Size([30, 3, 160, 160])
    target_img = data["center_img"] # torch.Size([30, 1, 160, 160])

    slice_ = data["slice_id"][0].zfill(3) # '000'
    direction = data["n_direction"][0].zfill(2) # '00' or '30', not the one in test_subject_paths
    cur_id = data["subject_id"][0] # 101309

    max_val = np.max(target_img)

    pred_imgs = torch.empty_like(target_img)

    for target_idx in range(30):
        
        # get the input image and target image
        input_imgs_singleDir = input_imgs[target_idx] 
        target_img_singleDir = target_img[target_idx]
        adjacent_direction = adjacent_directions[target_idx]
        center_direction = center_directions[target_idx]


        # get the interpolation image
        pred_img = interpolation_img_by_dir(center_direction, adjacent_direction, input_imgs_singleDir)
        # make it as a single channel image
        pred_img = pred_img.unsqueeze(0)
        # concatenate the image
        pred_imgs[target_idx] = pred_img

    target_img_cropped = transform_crop(target_img)
    pred_img_cropped = transform_crop(pred_imgs)

    SSIM_interp.append(ssim(pred_img_cropped, target_img_cropped).item())

    Path(f"/path-to-output-folder/results_n=6/{cur_id}").mkdir(parents=True, exist_ok=True)

    np.savez(
        f"/path-to-output-folder/results_n=6/{cur_id}/slice_{slice_}_direction_{direction}_SSIM_{SSIM_interp[-1]}", 
        predicted = pred_img_cropped.numpy(), 
        target = target_img_cropped.numpy() 
        )












       

    