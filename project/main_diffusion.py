import time
import glob
from argparse import ArgumentParser, Namespace
from pathlib import Path

import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import CenterCrop
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddpm import DDPMScheduler
from generative.networks.schedulers.ddim import DDIMScheduler

from utils.const import DATA_FOLDER, TENSORBOARD_LOG_DIR, MODEL_PARAMS_FOLDER
from utils.utils import seed_everything, get_train_val_test_subject_ids, get_data_path
from utils.utilization import get_config
from utils.transforms import get_train_transform, get_train_data_augmentation_transform
from utils.add_argument import add_argument
from utils.visualization import draw_gt_pred

torch.multiprocessing.set_sharing_strategy("file_system")


def main(hparams: Namespace, writer: SummaryWriter):
    train_subjects, val_subjects, test_subjects = get_train_val_test_subject_ids(
        DATA_FOLDER, train_percent=0.8, val_percent=0.1
    )

    print(f"Test subjects: {test_subjects}")

    train_subject_paths, val_subject_paths, test_subject_paths = (
        get_data_path(
            train_subjects,
            DATA_FOLDER,
            is_val=False,
            n_neighbor_directions=int(hparams.n_neighbor_directions),
        ),
        get_data_path(
            val_subjects,
            DATA_FOLDER,
            is_val=True,
            n_neighbor_directions=int(hparams.n_neighbor_directions),
        ),
        get_data_path(
            test_subjects,
            DATA_FOLDER,
            is_val=False,
            is_test=True,
            n_neighbor_directions=int(hparams.n_neighbor_directions),
        ),
    )

    if hparams.data_augmentation:
        train_transforms = get_train_data_augmentation_transform()
    else:
        train_transforms = get_train_transform()
    train_ds = Dataset(data=train_subject_paths, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=hparams.batch_size, shuffle=True, num_workers=8
    )

    # construct the val loader
    val_transforms = get_train_transform()
    val_ds = Dataset(data=val_subject_paths, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=36, shuffle=False, num_workers=8)

    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR / hparams.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    n_channels = [int(item) for item in str(hparams.n_channels).split(",")]
    n_in_channel = 1 + hparams.n_neighbor_directions
    cross_attention_dim = (1 + hparams.n_neighbor_directions) * 3

    # https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/diffusion_model_unet.py#L1646
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=n_in_channel,
        out_channels=1,
        num_channels=n_channels,
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=128,
        with_conditioning=True,
        cross_attention_dim=cross_attention_dim,
    )

    if hparams.load_model_training:
        model.load_state_dict(
            torch.load(
                MODEL_PARAMS_FOLDER
                / "diffusion_model_SSIM_0.8417819440364838_Diffusion_data_augmentation_p_1.pt"
            )
        )
    model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
    roi_size = (145, 174)
    transform_crop = CenterCrop(roi_size)
    n_epochs = 8
    val_interval = len(train_loader) // 5

    scaler = GradScaler()
    iteration, val_step = 0, 0
    highest_val_ssim = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        iteration += 1
        print(f"Epoch {epoch} started")

        for step, batch_data in enumerate(train_loader):
            adjacent_directions = (
                batch_data["adjacent_directions"].transpose(-2, -1).to(device)
            )  # [4, 3, n_neighbour_directions] -> [4, n_neighbour_directions, 3]

            center_directions = batch_data["center_directions"].to(device)
            input_directions = torch.cat(
                [adjacent_directions, center_directions.unsqueeze(1)], dim=1
            ).to(
                device
            )  # [4, n_neighbour_directions, 3]
            input_imgs = batch_data["adjacent_img"].to(device)
            target_img = batch_data["center_img"].to(device)

            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(input_imgs),)).to(device)

            with autocast(enabled=True):
                noise = torch.randn_like(target_img).to(device)
                noisy_img = scheduler.add_noise(
                    original_samples=target_img, noise=noise, timesteps=timesteps
                )
                combined = torch.cat((input_imgs, noisy_img), dim=1)
                # FIXME - we have a problem here: mat1 and mat2 shapes cannot be multiplied (14x3 and 21x64)
                # but if we remove conditioning, the model will work.
                # I now change it to input_directions.view(batch_size, 1, -1), which is of size
                # [batch_size, 1, 21]. It can run now! but i have no idea if this is the correct way to do it.

                prediction = model(
                    x=combined,
                    timesteps=timesteps,
                    context=input_directions.view(target_img.shape[0], 1, -1),
                )
                loss = F.mse_loss(prediction.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            writer.add_scalar("train_loss/loss", loss.item(), iteration)
            iteration += 1

            val_PSNRs, val_SSIMs = [], []
            if (step + 1) % val_interval == 0:
                val_step += 1
                start_time = time.time()
                for val_iter, data_val in enumerate(val_loader):
                    adjacent_directions = (
                        data_val["adjacent_directions"].transpose(-2, -1).to(device)
                    )
                    center_directions = data_val["center_directions"].to(device)
                    input_directions = torch.cat(
                        [adjacent_directions, center_directions.unsqueeze(1)], dim=1
                    ).to(
                        device
                    )  # [8, 7, 3]
                    input_imgs = data_val["adjacent_img"].to(device)
                    target_img = data_val["center_img"].to(device)
                    model.eval()

                    noise = torch.randn_like(target_img).to(device)
                    current_img = noise
                    combined = torch.cat((input_imgs, noise), dim=1)
                    scheduler.set_timesteps(num_inference_steps=1000)

                    for t in scheduler.timesteps:
                        with torch.no_grad():
                            with autocast(enabled=True):
                                model_output = model(
                                    combined,
                                    timesteps=torch.Tensor((t,)).to(current_img.device),
                                    context=input_directions.view(
                                        target_img.shape[0], 1, -1
                                    ),
                                )
                                current_img, _ = scheduler.step(
                                    model_output, t, current_img
                                )
                                combined = torch.cat((input_imgs, current_img), dim=1)

                    current_img_cropped = transform_crop(current_img)
                    target_img_cropped = transform_crop(target_img)

                    val_PSNRs.append(
                        psnr(current_img_cropped, target_img_cropped).item()
                    )
                    val_SSIMs.append(
                        ssim(current_img_cropped, target_img_cropped).item()
                    )

                    if val_iter == 1:
                        plt.figure("visualize", (8, 4))
                        plt.subplot(1, 2, 1)
                        plt.title("predict")
                        plt.imshow(
                            current_img_cropped[6, 0, :, :].detach().cpu().numpy(),
                            cmap="gray",
                        )
                        plt.axis("off")
                        plt.subplot(1, 2, 2)
                        plt.title("target")
                        plt.imshow(
                            target_img_cropped[6, 0, :, :].detach().cpu().numpy(),
                            cmap="gray",
                        )
                        plt.axis("off")
                        plt.tight_layout()
                        writer.add_figure(
                            f"Epoch: {epoch}, val_iter {val_step}", plt.gcf(), val_step
                        )

                writer.add_scalar("val_loss/val_PSNR", np.mean(val_PSNRs), val_step)
                writer.add_scalar("val_loss/val_SSIM", np.mean(val_SSIMs), val_step)
                # end time
                end_time = time.time()
                # report total time
                print("Time taken for validation: ", end_time - start_time)
                if np.mean(val_SSIMs) > highest_val_ssim:
                    highest_val_ssim = np.mean(val_SSIMs)
                    torch.save(
                        model.state_dict(),
                        f"/project/ace-ig/code/DWI/model_params/diffusion_model_SSIM_{np.mean(val_SSIMs)}_{hparams.experiment_name}.pt",
                    )
                    print(f"Model saved with SSIM: {np.mean(val_SSIMs)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR / hparams.experiment_name)
    main(hparams, writer)
