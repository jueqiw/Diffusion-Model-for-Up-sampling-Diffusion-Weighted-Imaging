import os
import re
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

from utils.const import DATA_FOLDER


def get_train_val_test_subject_ids(
    folder: Path, train_percent: float, val_percent: float
) -> Tuple[set, set, set]:
    all_files = folder.glob("*.npz")

    idxs = []
    for item in all_files:
        pattern = r"(\d+)_"
        match = re.search(pattern, str(item))
        result = match.group(1)
        idxs.append(result)

    idxs = sorted(list(set(idxs)))
    random.shuffle(idxs)
    train_size = int(len(idxs) * train_percent)
    val_size = int(len(idxs) * val_percent)

    train_ids = sorted(list(idxs[:train_size]))
    val_ids = sorted(list(idxs[train_size : train_size + val_size]))
    test_ids = sorted(list(idxs[train_size + val_size :]))

    return train_ids, val_ids, test_ids


def get_data_path(
    subject_ids: List[str],
    folder: Path,
    is_val: bool,
    is_test: bool = False,
    n_neighbor_directions: int = 6,
) -> List[dict]:
    sample_dicts = []

    for subject_id in subject_ids:
        if is_val:
            adjacent_slice_path = folder / f"{subject_id}_slice74_ref.npz"
            center_slice_path = folder / f"{subject_id}_slice74_rem.npz"
            adjacent_direction_path = folder / f"{subject_id}_bvec74_ref.npz"
            center_direction_path = folder / f"{subject_id}_bvec74_rem.npz"
            sample_dicts.append(
                {
                    "adjacent_img": str(adjacent_slice_path),
                    "adjacent_direction": str(adjacent_direction_path),
                    "center_img": str(center_slice_path),
                    "center_direction": str(center_direction_path),
                    "subject_id": subject_id,
                    "slice_id": "74",
                    "n_neighbor_directions": f"{n_neighbor_directions}",
                }
            )
            continue

        for idx in range(0, 145):
            adjacent_slice_path = folder / f"{subject_id}_slice{idx}_ref.npz"
            center_slice_path = folder / f"{subject_id}_slice{idx}_rem.npz"
            adjacent_direction_path = folder / f"{subject_id}_bvec{idx}_ref.npz"
            center_direction_path = folder / f"{subject_id}_bvec{idx}_rem.npz"

            if str(center_slice_path) == str(DATA_FOLDER / "185341_slice13_rem.npz"):
                continue

            if is_test:
                for direction in range(0, 60):
                    sample_dicts.append(
                        {
                            "adjacent_img": str(adjacent_slice_path),
                            "adjacent_direction": str(adjacent_direction_path),
                            "center_img": str(center_slice_path),
                            "center_direction": str(center_direction_path),
                            "subject_id": subject_id,
                            "slice_id": f"{idx}",
                            "n_direction": f"{direction}",
                            "n_neighbor_directions": f"{n_neighbor_directions}",
                        }
                    )
                continue

            sample_dicts.append(
                {
                    "adjacent_img": str(adjacent_slice_path),
                    "adjacent_direction": str(adjacent_direction_path),
                    "center_img": str(center_slice_path),
                    "center_direction": str(center_direction_path),
                    "subject_id": subject_id,
                    "slice_id": f"{idx}",
                    "n_neighbor_directions": f"{n_neighbor_directions}",
                }
            )

    return sample_dicts


def set_requires_grad(nets, requires_grad=False):
    """
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
