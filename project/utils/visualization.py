import numpy as np
import matplotlib.pyplot as plt
import torch

# import cv2
import time
import sys


def tensorboard_vis(
    summarywriter, step, board_name, img_list, num_row, cmaps, titles, resize=True
):
    """
    :param summarywriter: tensorboard summary writer handle
    :param step:
    :param board_name: display name
    :param img_list: a list of images to show
    :param num_row: specify the number of row
    :param cmaps: specify color maps for each image ('gray' for MRI, i.e.)
    :param titles: specify each image title
    :param resize: whether resize the image to show
    :return:
    """
    fig = plt.figure()
    num_figs = len(img_list)
    num_col = int(np.ceil(num_figs / num_row))
    print("Visualizing %d images in %d row %d column" % (num_figs, num_row, num_col))

    for i in range(num_figs):
        ax = plt.subplot(num_row, num_col, i + 1)
        tmp = img_list[i]
        if isinstance(cmaps, str):
            c = cmaps
        else:
            c = cmaps[i]
        vmin = tmp.min()
        vmax = tmp.max()
        ax.imshow(tmp, cmap=c, vmin=vmin, vmax=vmax), plt.title(titles[i]), plt.axis(
            "off"
        )

    summarywriter.add_figure(board_name, fig, step)
    return summarywriter


def create_group_fig(
    img_list,
    num_row,
    cmaps,
    titles,
    resize=True,
    save_name=None,
    fig_size=[15, 15],
    vmin=None,
    vmax=None,
    dpi=100,
    format="eps",
):
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()
    num_figs = len(img_list)
    num_col = int(np.ceil(num_figs / num_row))
    # print('Visualizing %d images in %d row %d column' % (num_figs, num_row, num_col))
    for i in range(num_figs):
        ax = plt.subplot(num_row, num_col, i + 1)
        tmp = img_list[i]
        if isinstance(cmaps, str):
            c = cmaps
        else:
            c = cmaps[i]

        if vmax is not None:
            if isinstance(vmax, list):
                v_min, v_max = vmin[i], vmax[i]
            else:
                v_min, v_max = vmin, vmax
        else:
            v_min, v_max = tmp.min(), tmp.max()
        ax.imshow(tmp, cmap=c, vmin=v_min, vmax=v_max), plt.title(titles[i]), plt.axis(
            "off"
        )

    if save_name:
        s = time.time()
        if dpi:
            plt.savefig(save_name, format=format, dpi=dpi)
        else:
            plt.savefig(save_name, format=format)
        e = time.time()

    return fig


def draw_gt_pred(gt, pred, save_file=None, vmin=None, vmax=None, dpi=300, format="eps"):
    plt.rcParams["figure.figsize"] = [8, 4]
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Prediction")
    plt.axis("off")

    if save_file:
        if dpi:
            plt.savefig(save_file, format=format, dpi=dpi)
        else:
            plt.savefig(save_file, format=format)

    return fig


def save_pred_img(
    gt: np.ndarray,
    GAN_pred: np.ndarray,
    Diffusion_pred: np.ndarray,
    save_path: str,
    SSIM_GAN: float,
    SSIM_diffusion: float,
):
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(GAN_pred, cmap="gray")
    plt.title("GAN Prediction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(Diffusion_pred, cmap="gray")
    plt.title("Diffusion Prediction")
    plt.axis("off")

    # tight_layout
    plt.tight_layout()
    # set overall title for the figure
    plt.suptitle(f"SSIM GAN: {SSIM_GAN:.4f}, SSIM Diffusion: {SSIM_diffusion:.4f}")
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    sys.exit()
