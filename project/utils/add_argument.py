from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--model",
        choices=["GAN", "diffusion", "qGAN", "interp"],
        type=str,
    )
    parser.add_argument(
        "--experiment_name",
        default="Diffusion",
        help="Experiment name for TensorBoardLogger",
    )
    parser.add_argument("--start_id", default=0, type=int)
    parser.add_argument("--end_id", default=2, type=int)
    parser.add_argument(
        "--n_channels",
        default="128,128,256",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--disc_start_iter_each_time",
        default=10000000000000000000000000,
        type=int,
    )
    parser.add_argument(
        "--l1_weight",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--data_augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--load_model_training",
        action="store_true",
    )
    parser.add_argument(
        "--n_neighbor_directions",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--load_model_name",
        type=str,
    )
    parser.add_argument(
        "--save_folder",
        type=str,
    )
