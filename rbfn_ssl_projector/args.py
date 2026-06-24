import argparse

from argparse import Namespace


def get_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Radial Basis Function Networks as Projection Heads in Self-Supervised Learning."
    )


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Hardware
    parser.add_argument(
        "--devices",
        type=str,
        default="all",
        help="Comma separated list of device (GPUs) to use or '-1' for all.",
    )

    # Dataset
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data",
        help="Download path location of dataset to use.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[
            "ImageNet100",
            "OpenImagesV7-50",
            "OpenImagesV7-30",
            "OpenImagesV7-10",
        ],
        default="OpenImagesV7-50",
        help="Dataset name to use for training and evaluation.",
    )
    parser.add_argument(
        "--train-perc",
        type=float,
        default=0.7,
        help="Train split percentage",
    )
    parser.add_argument(
        "--val-perc",
        type=float,
        default=0.1,
        help="Validation split percentage",
    )
    parser.add_argument(
        "--test-perc",
        type=float,
        default=0.2,
        help="Test split percentage",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Square image size in pixels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size.",
    )
    parser.add_argument(
        "--normalize",
        default=False,
        action="store_const",
        const=True,
        help="Normalize dataset splits using Z-score normalization.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory",
        default=False,
        action="store_const",
        const=True,
        help="Pin GPU memory",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="Datatype for loading dataset",
    )

    # SSL architecture
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["simclr", "simsiam", "moco", "byol", "swav"],
        default="simclr",
        help="SSL architecture",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet34",
        help="Feature extraction backbone",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=["DEFAULT", "IMAGENET1K_V1"],
        default=None,
        help="Pretrained weights to use for feature extraction backbone.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["SGD", "Adam", "AdamW", "LARS"],
        help="Optimization algorithm used while training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        choices=["ReduceLROnPlateau", "StepLR", "ExponentialLR", "CosineAnnealingLR"],
        help="Learning rate scheduler used while training.",
    )
    parser.add_argument(
        "--base-learning-rate",
        type=float,
        default=None,
        help="Base learning rate used to calculate actual learning rate based on batch size (linear scaling rule).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate to use while training. Overrides learning rate calculation via linear scaling rule.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=None,
        help="Temperature parameter used within NTXentLoss contrastive loss.",
    )
    parser.add_argument(
        "--lr-scheduler-gamma",
        type=float,
        default=None,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=None,
        help="Period of learning rate decay used while training with StepLR learning rate scheduler.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Momentum for SGD optimizer.",
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        default=None,
        help="Save top k models models only instead of all or the last.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs to wait for validation loss to improve.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs to train for.",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory used for logging.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name used for trainig experiment.",
    )
    parser.add_argument(
        "--experiment-eval-name",
        type=str,
        required=True,
        help="Name used for evaluation experiment.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name used for the run",
    )
    parser.add_argument(
        "--use-mlflow",
        default=False,
        action="store_const",
        const=True,
        help="Use MLFlow for experiment tracking.",
    )

    # Projection head
    parser.add_argument(
        "--projection-head-type",
        type=str,
        choices=["default", "rbfn"],
        default="default",
        help="Type of projection head to use for architecture.",
    )
    parser.add_argument(
        "--projection-head-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimensions of projection head. If None, same as the number of backbone's output features.",
    )
    parser.add_argument(
        "--projection-head-output-dim",
        type=int,
        default=128,
        help="Output dimensions of projection head.",
    )
    parser.add_argument(
        "--projection-head-num-layers",
        type=int,
        default=2,
        help="Number of layers in projection head.",
    )
    parser.add_argument(
        "--projection-head-batch-norm",
        default=None,
        action="store_const",
        const=True,
        help="Apply batch normalization in projection head.",
    )

    ## RBFN projection head
    parser.add_argument(
        "--rbfn-projection-head-num-kernels",
        type=int,
        default=None,
        help="Number of kernels to use within RBFN projection head.",
    )
    parser.add_argument(
        "--rbfn-projection-head-radial-function",
        type=str,
        choices=[
            "linear",
            "gaussian",
            "inverse_quadratic",
            "inverse_multiquadric",
            "multiquadric",
            "rth",
            "tps",
        ],
        default=None,
        help="Radial basis function to use for RBFN projection head.",
    )
    parser.add_argument(
        "--rbfn-projection-head-norm-function",
        type=str,
        choices=["euclidian", "manhattan"],
        default=None,
        help="Distance function to use for RBFN projection head.",
    )
    parser.add_argument(
        "--rbfn-projection-head-normalize",
        default=False,
        action="store_const",
        const=True,
        help="Use a normalized RBFN architecture as projection head.",
    )

    # Evaluation
    parser.add_argument(
        "--eval-train-perc",
        type=float,
        default=0.7,
        help="Percentage from evaluation dataset to use for training.",
    )
    parser.add_argument(
        "--eval-val-perc",
        type=float,
        default=0.1,
        help="Percentage from evaluation dataset to use for validation.",
    )
    parser.add_argument(
        "--eval-test-perc",
        type=float,
        default=0.2,
        help="Percentage from evaluation dataset to use for testing.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=100,
        help="Number of epochs to train logistic regression for.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--eval-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate used while training logistic regression.",
    )
    parser.add_argument(
        "--eval-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used while training logistic regression.",
    )
    parser.add_argument(
        "--eval-patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs to wait for validation loss to improve while evaluating trained SSL model.",
    )

    return parser


def get_args() -> Namespace:
    parser = get_parser()
    parser = add_args(parser)
    return parser.parse_args()
