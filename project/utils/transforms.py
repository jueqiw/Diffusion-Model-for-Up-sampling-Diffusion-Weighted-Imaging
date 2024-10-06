from typing import Dict, Tuple

import torch
import numpy as np
from monai.data import NumpyReader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    SpatialCropd,
    SpatialPadd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    MapTransform,
    RandRotated,
    RandZoomd,
    RandAffined,
)


def sample_directions_training(
    adjacent_bvec: np.ndarray, center_bvec: np.ndarray, n_neighbor_directions: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    n_center_directions = center_bvec.shape[1]  # shape 3, 60
    num_centers = 1

    center_indices = np.random.choice(
        n_center_directions, size=num_centers, replace=False
    )

    center_vec = center_bvec[:, center_indices[0]]
    similarity = np.dot(adjacent_bvec.T, center_vec)

    sorted_indices = np.argsort(-similarity)
    adjacent_indices = sorted_indices[:n_neighbor_directions]

    return center_indices[0], adjacent_indices


def sample_directions_testing(
    adjacent_bvec: np.ndarray,
    center_bvec: np.ndarray,
    n_direction: int,
    n_neighbor_directions: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    center_vec = center_bvec[:, n_direction]
    similarity = np.dot(adjacent_bvec.T, center_vec)

    sorted_indices = np.argsort(-similarity)
    adjacent_indices = sorted_indices[:n_neighbor_directions]

    return n_direction, adjacent_indices


class GetCenterAdjacentImgTest(MapTransform):
    def __init__(self, keys):
        MapTransform.__init__(self, keys)

    def __call__(self, x: Dict):
        center_indices, adjacent_indices = sample_directions_testing(
            x["adjacent_direction"],
            x["center_direction"],
            int(x["n_direction"]),
            n_neighbor_directions=int(x["n_neighbor_directions"]),
        )
        center_image = x["center_img"][center_indices, :, :]
        adjacent_images = x["adjacent_img"][adjacent_indices, :, :]
        center_direction = x["center_direction"][:, center_indices]
        adjacent_directions = x["adjacent_direction"][:, adjacent_indices]

        sample = {
            "adjacent_img": adjacent_images,
            "adjacent_directions": adjacent_directions,
            # create channel dimension
            "center_img": center_image.unsqueeze(0),
            "center_directions": center_direction,
            "subject_id": x["subject_id"],
            "slice_id": x["slice_id"],
            "n_direction": x["n_direction"],
        }

        return sample


class GetCenterAdjacentImg(MapTransform):
    def __init__(self, keys):
        MapTransform.__init__(self, keys)

    def __call__(self, x: Dict):
        center_indices, adjacent_indices = sample_directions_training(
            x["adjacent_direction"],
            x["center_direction"],
            int(x["n_neighbor_directions"]),
        )
        center_image = x["center_img"][center_indices, :, :]
        adjacent_images = x["adjacent_img"][adjacent_indices, :, :]
        center_direction = x["center_direction"][:, center_indices]
        adjacent_directions = x["adjacent_direction"][:, adjacent_indices]

        sample = {
            "adjacent_img": adjacent_images,
            "adjacent_directions": adjacent_directions,
            # create channel dimension
            "center_img": center_image.unsqueeze(0),
            "center_directions": center_direction,
            "subject_id": x["subject_id"],
            "slice_id": x["slice_id"],
        }

        return sample


def get_train_data_augmentation_transform() -> Compose:
    return Compose(
        [
            LoadImaged(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ],
                reader=NumpyReader(),
            ),
            GetCenterAdjacentImg(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ]
            ),
            ScaleIntensityd(
                keys=[
                    "adjacent_img",
                    "center_img",
                ],
                minv=0.0,
                maxv=1.0,
            ),
            RandSpatialCropd(
                keys=["adjacent_img", "center_img"],
                roi_size=[118, 153],
            ),
            RandAffined(
                keys=["adjacent_img", "center_img"],
                rotate_range=[-15, 15],
                scale_range=[0.9, 1.1],
                prob=1,
            ),
            SpatialPadd(
                keys=["adjacent_img", "center_img"],
                spatial_size=[160, 160],
            ),
        ]
    )


def get_train_transform() -> Compose:
    return Compose(
        [
            LoadImaged(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ],
                reader=NumpyReader(),
            ),
            GetCenterAdjacentImg(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ]
            ),
            ScaleIntensityd(
                keys=[
                    "adjacent_img",
                    "center_img",
                ],
                minv=0.0,
                maxv=1.0,
            ),
            SpatialCropd(
                keys=["adjacent_img", "center_img"],
                roi_start=[14, 10],
                roi_end=[132, 163],
            ),
            SpatialPadd(
                keys=["adjacent_img", "center_img"],
                spatial_size=[160, 160],
            ),
        ]
    )


def get_test_transform() -> Compose:
    return Compose(
        [
            LoadImaged(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ],
                reader=NumpyReader(),
            ),
            GetCenterAdjacentImgTest(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ]
            ),
            ScaleIntensityd(
                keys=[
                    "adjacent_img",
                    "center_img",
                ],
                minv=0.0,
                maxv=1.0,
            ),
            SpatialCropd(
                keys=["adjacent_img", "center_img"],
                roi_start=[14, 10],
                roi_end=[132, 163],
            ),
            SpatialPadd(
                keys=["adjacent_img", "center_img"],
                spatial_size=[160, 160],
            ),
        ]
    )


def get_test_transform_no_scaling() -> Compose:
    return Compose(
        [
            LoadImaged(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ],
                reader=NumpyReader(),
            ),
            GetCenterAdjacentImgTest(
                keys=[
                    "adjacent_img",
                    "adjacent_direction",
                    "center_img",
                    "center_direction",
                ]
            ),
            # ScaleIntensityd(
            #     keys=[
            #         "adjacent_img",
            #         "center_img",
            #     ],
            #     minv=0.0,
            #     maxv=1.0,
            # ),
            SpatialCropd(
                keys=["adjacent_img", "center_img"],
                roi_start=[14, 10],
                roi_end=[132, 163],
            ),
            SpatialPadd(
                keys=["adjacent_img", "center_img"],
                spatial_size=[160, 160],
            ),
        ]
    )
