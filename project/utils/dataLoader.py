import numpy as np


def sample_directions(bvec, idx):
    np.random.seed(idx)  # fixed
    num_directions = bvec.shape[1]  # Should be 90
    num_centers = 9
    adjacents_per_center = 6

    # Randomly select 9 center indices
    center_indices = np.random.choice(num_directions, size=num_centers, replace=False)
    adjacent_indices = []

    for center_idx in center_indices:
        center_vec = bvec[:, center_idx]

        # compute distance
        dist = np.dot(bvec.T, center_vec)

        sorted_indices = np.argsort(-dist)
        # filtered_indices = [idx for idx in sorted_indices if idx not in center_indices and idx not in adjacent_indices][:adjacents_per_center]
        filtered_indices = [idx for idx in sorted_indices if idx not in center_indices][
            :adjacents_per_center
        ]
        adjacent_indices.extend(filtered_indices)

    return list(center_indices), adjacent_indices


class HCPdiffusion:
    def __init__(self, indices):
        self.indices = indices
        self.data, self.bvecs, self.subject_ids = self.load_data()

    def load_data(self):
        slices = [
            np.load(f"/projectnb/dl523/projects/DWI/data/HCPdiff_slice{i+1}.npy")[
                self.indices
            ]
            for i in range(6)
        ]
        bvecs = [
            np.load(f"/projectnb/dl523/projects/DWI/data/HCPdiff_bvec{i+1}.npy")[
                self.indices
            ]
            for i in range(6)
        ]

        # Concatenate slices and vectors across the 6 files for each subject
        # for example, if indices=[2,3,4], then we are stacking like [2,3,4,2,3,4,...]
        concatenated_slices = np.concatenate(slices, axis=0)
        concatenated_bvecs = np.concatenate(bvecs, axis=0)
        subject_ids = np.tile(self.indices, 6)

        return concatenated_slices, concatenated_bvecs, subject_ids

    def __len__(self):
        return len(self.indices) * 6 * 9

    def __getitem__(self, idx):
        # idx is a scalar!
        # Since __len__(self) returns N*6*9, where N is the trainset/valset/testset size
        # idx knows that it can go from 0 to N*6*9-1 to produce a single sample

        # self.data.shpae: [N*6, 90, 145, 174]
        # self.bvecs.shpae: [N*6, 3, 90]

        # need to convert idx (i.e., sample idx) to slice_idx
        # for example: when we pick idx = 82 (the 82th sample), we are refering to
        # the 9th slice (slice_idx: 0~N*6) and the 1st sample in that slice (pos_in_slice: 0~8)

        slice_idx = idx // 9
        pos_in_slice = idx % 9

        slice_data = self.data[slice_idx]  # shape: (90, 145, 174)
        slice_bvec = self.bvecs[slice_idx]  # shape: (3,90)

        center_indices, adjacent_indices = sample_directions(slice_bvec, slice_idx)
        # center_indices: 9x1 list
        # adjacent_indices: 54x1 list, 0~5 correspond to center_indices[0],
        #                              6~11 correspond to center_indices[1], etc.

        # according to the position in slice, pick the center image
        center_image = slice_data[center_indices[pos_in_slice], :, :]

        # according to the position in slice, pick 6 corresponding adjacent images
        adjacent_images = slice_data[
            adjacent_indices[pos_in_slice * 6 : (pos_in_slice + 1) * 6], :, :
        ]

        # similarly
        center_direction = slice_bvec[:, center_indices[pos_in_slice]]
        adjacent_directions = slice_bvec[
            :, adjacent_indices[pos_in_slice * 6 : (pos_in_slice + 1) * 6]
        ]

        subject_id = self.subject_ids[slice_idx]

        subsample = {
            "center": {"images": center_image, "directions": center_direction},
            "adjacent": {"images": adjacent_images, "directions": adjacent_directions},
            "subjectid": subject_id,
        }

        return subsample


################# example #################
# say, the trainset contains subject [1,2,3]
# >> trainset = HCPdiffusion([1,2,3])
# >> import torch
# >> trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
# >> dataiter = iter(trainloader)
# >> subsample = next(dataiter)
# There will be 3*6*9=162 subsamples in this case. You can use the following to verify
# >> print(sum(1 for _ in trainloader))
###########################################
