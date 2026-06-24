import os
import torch
import rbfn_head
import torchvision
import lightning as L

from torch import nn
from pathlib import Path
from argparse import Namespace
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import Logger, CSVLogger, MLFlowLogger

from rbfn_ssl_projector.enums import (
    ProjectionHead as ProjectionHeadType,
    Architecture as ArchitectureType,
)

# Datamodule imports
from simclr.datamodule import SimCLRDataModule
from simsiam.datamodule import SimSiamDataModule
from moco.datamodule import MoCoDataModule
from byol.datamodule import BYOLDataModule
from swav.datamodule import SwaVDataModule

# Projection head imports
from rbfn_head.head import RBFNProjectionHead, RBFNPredictionHead
from lightly.models.modules.heads import ProjectionHead
from lightly.models.modules.heads import (
    SimCLRProjectionHead,
    SimSiamProjectionHead,
    SimSiamPredictionHead,
    BYOLProjectionHead,
    BYOLPredictionHead,
    SwaVProjectionHead,
    MoCoProjectionHead,
)

# Architecture module imports
from simclr.module import SimCLR
from moco.module import MoCo
from simsiam.module import SimSiam
from byol.module import BYOL
from swav.module import SwaV


def seed_rng(seed: int = 42) -> None:
    seed_everything(seed=seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def print_gpu_info() -> None:
    print("\nGPU info:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"   Memory: {props.total_memory // 1e9} GB")
        print(f"   Compute capability: {props.major}.{props.minor}")
        print(f"   Multi-processor count: {props.multi_processor_count}")


def create_loggers(
    log_dir: str,
    experiment_name: str,
    run_name: str,
    use_mlflow: bool = False,
) -> dict[str, Logger]:
    loggers = {
        "CSVLogger": CSVLogger(
            save_dir=str(Path(log_dir, "csv", experiment_name)),
            name=run_name,
        )
    }

    if use_mlflow:
        loggers["MLFlowLogger"] = MLFlowLogger(
            save_dir=str(Path(log_dir, "mlruns", experiment_name)),
            experiment_name=experiment_name,
            run_name=run_name,
            log_model=True,
        )

    return loggers


def create_trainer_callbacks(
    patience: int | None, save_top_k: int | None
) -> dict[str, Callback]:
    trainer_callbacks = {}
    if patience is not None:
        trainer_callbacks["EarlyStopping"] = EarlyStopping(
            patience=patience, monitor="val_loss"
        )
    if save_top_k is not None:
        trainer_callbacks["ModelCheckpoint"] = ModelCheckpoint(
            save_top_k=save_top_k,
            auto_insert_metric_name=True,
            save_last=patience is None,
            monitor="val_loss",
        )
    return trainer_callbacks


def get_parameter_stats(module: L.LightningModule, prefix: str = "") -> dict:
    return {
        f"{prefix}trainable_params": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
        f"{prefix}non_trainable_params": sum(
            p.numel() for p in module.parameters() if not p.requires_grad
        ),
    }


def get_datamodule(args: Namespace, logger: list[Logger]) -> L.LightningDataModule:
    match args.architecture:
        case ArchitectureType.SIMCLR:
            DataModule = SimCLRDataModule
        case ArchitectureType.SIMSIAM:
            DataModule = SimSiamDataModule
        case ArchitectureType.MOCO:
            DataModule = MoCoDataModule
        case ArchitectureType.BYOL:
            DataModule = BYOLDataModule
        case ArchitectureType.SWAV:
            DataModule = SwaVDataModule
        case _:
            raise ValueError(f"Unsupported architecture: {args.architecture}")

    return DataModule(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        train_perc=args.train_perc,
        val_perc=args.val_perc,
        test_perc=args.test_perc,
        img_size=args.img_size,
        batch_size=args.batch_size,
        normalize=args.normalize,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        precision=(
            getattr(torch, args.precision) if args.precision is not None else None
        ),
        logger=logger,
    )


def get_backbone_model(backbone: str, weights: str | None) -> nn.Module:
    return getattr(torchvision.models, backbone)(weights=weights)


def has_prediction_head(architecture: ArchitectureType) -> bool:
    return architecture in {ArchitectureType.SIMSIAM, ArchitectureType.BYOL}


def get_heads(
    args: Namespace, backbone_model: nn.Module
) -> tuple[ProjectionHead, ProjectionHead | None, dict[str, int | None]]:

    def _get_projection_head() -> ProjectionHead:
        num_backbone_output_features = list(backbone_model.children())[-1].in_features

        if args.projection_head_hidden_dim is None:
            projection_head_hidden_dim = num_backbone_output_features
        else:
            projection_head_hidden_dim = args.projection_head_hidden_dim

        projection_head_kwargs = {
            "args": args,
            "projection_head_input_dim": num_backbone_output_features,
            "projection_head_hidden_dim": projection_head_hidden_dim,
            "projection_head_output_dim": args.projection_head_output_dim,
        }

        match args.projection_head_type:
            case ProjectionHeadType.RBFN:
                projection_head, params = get_rbfn_projection_head(
                    **projection_head_kwargs
                )
            case ProjectionHeadType.DEFAULT:
                projection_head, params = get_default_projection_head(
                    **projection_head_kwargs
                )
            case _:
                raise ValueError(
                    "Unsupported projection head type:", args.projection_head_type
                )
        return projection_head, params

    def _get_prediction_head() -> tuple[ProjectionHead | None, dict[str, int | None]]:
        prediction_head, params = None, {}
        if has_prediction_head(args.architecture):
            prediction_head_kwargs = {
                "args": args,
                "prediction_head_input_dim": args.projection_head_output_dim,
            }
            match args.architecture:
                case ArchitectureType.BYOL:
                    # For '* 16' see: https://github.com/lightly-ai/lightly/blob/ee30cd481d68862c80de4ef45920cfe1ab1f67b1/lightly/models/modules/heads.py#L133
                    prediction_head_kwargs |= {
                        "prediction_head_hidden_dim": args.projection_head_output_dim
                        * 16,
                        "prediction_head_output_dim": args.projection_head_output_dim,
                    }
                case ArchitectureType.SIMSIAM:
                    # For '// 4' see: https://github.com/lightly-ai/lightly/blob/ee30cd481d68862c80de4ef45920cfe1ab1f67b1/lightly/models/modules/heads.py#L511
                    prediction_head_kwargs |= {
                        "prediction_head_hidden_dim": args.projection_head_output_dim
                        // 4,
                        "prediction_head_output_dim": args.projection_head_output_dim,
                    }
                case _:
                    raise ValueError(
                        f"Architecture '{args.architecture}' does not feature a prediction head."
                    )

            match args.projection_head_type:
                case ProjectionHeadType.RBFN:
                    prediction_head, params = get_rbfn_prediction_head(
                        **prediction_head_kwargs
                    )
                case ProjectionHeadType.DEFAULT:
                    prediction_head, params = get_default_prediction_head(
                        **prediction_head_kwargs
                    )
                case _:
                    raise ValueError(
                        "Unsupported prediction head type:", args.prediction_head_type
                    )
        return prediction_head, params

    projection_head, projection_head_params = _get_projection_head()
    prediction_head, prediction_head_params = _get_prediction_head()

    return (
        projection_head,
        prediction_head,
        (projection_head_params | prediction_head_params),
    )


def get_default_projection_head(
    args: Namespace,
    projection_head_input_dim: int,
    projection_head_hidden_dim: int,
    projection_head_output_dim: int,
) -> tuple[ProjectionHead, dict[str, int | None]]:
    match args.architecture:
        case ArchitectureType.SIMCLR:
            projection_head = SimCLRProjectionHead(
                input_dim=projection_head_input_dim,
                hidden_dim=projection_head_hidden_dim,
                output_dim=projection_head_output_dim,
                num_layers=args.projection_head_num_layers,
                batch_norm=args.projection_head_batch_norm,
            )
        case ArchitectureType.MOCO:
            projection_head = MoCoProjectionHead(
                input_dim=projection_head_input_dim,
                hidden_dim=projection_head_hidden_dim,
                output_dim=projection_head_output_dim,
                num_layers=args.projection_head_num_layers,
                batch_norm=args.projection_head_batch_norm,
            )
        case ArchitectureType.SIMSIAM:
            projection_head = SimSiamProjectionHead(
                input_dim=projection_head_input_dim,
                hidden_dim=projection_head_hidden_dim,
                output_dim=projection_head_output_dim,
            )
        case ArchitectureType.BYOL:
            projection_head = BYOLProjectionHead(
                input_dim=projection_head_input_dim,
                hidden_dim=projection_head_hidden_dim,
                output_dim=projection_head_output_dim,
            )
        case ArchitectureType.SWAV:
            projection_head = SwaVProjectionHead(
                input_dim=projection_head_input_dim,
                hidden_dim=projection_head_hidden_dim,
                output_dim=projection_head_output_dim,
            )
        case _:
            raise ValueError(
                f"Cannot build default projection head for unsupported architecture '{args.architecture}'."
            )

    params = {
        "projection_head_input_dim": projection_head_input_dim,
        "projection_head_hidden_dim": projection_head_hidden_dim,
        "projection_head_output_dim": projection_head_output_dim,
    }

    return projection_head, params


def get_rbfn_projection_head(
    args: Namespace,
    projection_head_input_dim: int,
    projection_head_hidden_dim: int,
    projection_head_output_dim: int,
) -> tuple[RBFNProjectionHead, int]:
    if args.rbfn_projection_head_num_kernels is None:
        rbfn_projection_head_num_kernels = projection_head_hidden_dim
    else:
        rbfn_projection_head_num_kernels = args.rbfn_projection_head_num_kernels

    rbfn_projection_head = RBFNProjectionHead(
        input_dim=projection_head_input_dim,
        hidden_dim=projection_head_hidden_dim,
        output_dim=projection_head_output_dim,
        num_kernels=rbfn_projection_head_num_kernels,
        radial_function=getattr(
            rbfn_head.radials, args.rbfn_projection_head_radial_function
        ),
        norm_function=getattr(rbfn_head.norms, args.rbfn_projection_head_norm_function),
        normalization=args.rbfn_projection_head_normalize,
        num_layers=args.projection_head_num_layers,
        batch_norm=args.projection_head_batch_norm,
    )

    params = {
        "projection_head_input_dim": projection_head_input_dim,
        "projection_head_hidden_dim": projection_head_hidden_dim,
        "projection_head_output_dim": projection_head_output_dim,
        "rbfn_projection_head_num_kernels": rbfn_projection_head_num_kernels,
    }

    return rbfn_projection_head, params


def get_default_prediction_head(
    args: Namespace,
    prediction_head_input_dim: int,
    prediction_head_hidden_dim: int,
    prediction_head_output_dim: int,
) -> tuple[ProjectionHead, dict[str, int | None]]:
    match args.architecture:
        case ArchitectureType.BYOL:
            default_prediction_head = BYOLPredictionHead(
                input_dim=prediction_head_input_dim,
                hidden_dim=prediction_head_hidden_dim,
                output_dim=prediction_head_output_dim,
            )
        case ArchitectureType.SIMSIAM:
            default_prediction_head = SimSiamPredictionHead(
                input_dim=prediction_head_input_dim,
                hidden_dim=prediction_head_hidden_dim,
                output_dim=prediction_head_output_dim,
            )
        case _:
            raise ValueError(
                f"Cannot build default prediction head for unsupported architecture '{args.architecture}'."
            )

    params = {
        "prediction_head_input_dim": prediction_head_input_dim,
        "prediction_head_hidden_dim": prediction_head_hidden_dim,
        "prediction_head_output_dim": prediction_head_output_dim,
    }

    return default_prediction_head, params


def get_rbfn_prediction_head(
    args: Namespace,
    prediction_head_input_dim: int,
    prediction_head_hidden_dim: int,
    prediction_head_output_dim: int,
) -> tuple[RBFNPredictionHead, dict[str, int | None]]:
    if args.rbfn_projection_head_num_kernels is None:
        rbfn_prediction_head_num_kernels = prediction_head_hidden_dim
    else:
        rbfn_prediction_head_num_kernels = args.rbfn_projection_head_num_kernels

    rbfn_prediction_head = RBFNPredictionHead(
        input_dim=prediction_head_input_dim,
        hidden_dim=prediction_head_hidden_dim,
        output_dim=prediction_head_output_dim,
        num_kernels=rbfn_prediction_head_num_kernels,
        radial_function=getattr(
            rbfn_head.radials, args.rbfn_projection_head_radial_function
        ),
        norm_function=getattr(rbfn_head.norms, args.rbfn_projection_head_norm_function),
        normalization=args.rbfn_projection_head_normalize,
    )

    params = {
        "prediction_head_input_dim": prediction_head_input_dim,
        "prediction_head_hidden_dim": prediction_head_hidden_dim,
        "prediction_head_output_dim": prediction_head_output_dim,
        "rbfn_prediction_head_num_kernels": rbfn_prediction_head_num_kernels,
        "rbfn_prediction_head_radial_function": args.rbfn_projection_head_radial_function,
        "rbfn_prediction_head_norm_function": args.rbfn_projection_head_norm_function,
        "rbfn_prediction_head_normalization": args.rbfn_projection_head_normalize,
    }

    return rbfn_prediction_head, params


def get_learning_rate(
    architecture: ArchitectureType,
    base_learning_rate: float | None,
    learning_rate: float | None,
    batch_size: int,
) -> tuple[float, float]:
    if base_learning_rate is None:
        match architecture:
            case ArchitectureType.SIMCLR:
                base_learning_rate = 0.3
            case ArchitectureType.MOCO:
                base_learning_rate = 0.03
            case ArchitectureType.SIMSIAM:
                base_learning_rate = 0.05
            case ArchitectureType.BYOL:
                base_learning_rate = 0.2
            case ArchitectureType.SWAV:
                base_learning_rate = 0.6
            case _:
                raise ValueError(f"Unsupported architecture: {architecture}")

    assert (
        base_learning_rate is not None
    ), f"Unable to determine learning rate for architecture {architecture}."

    learning_rate = (
        base_learning_rate * (batch_size / 256)
        if learning_rate is None
        else learning_rate
    )
    return base_learning_rate, learning_rate


def get_module(
    args: Namespace,
    backbone_model: nn.Module,
    projection_head: ProjectionHead,
    prediction_head: ProjectionHead | None,
    projection_head_hidden_dim: int,
    learning_rate: float,
) -> L.LightningModule:
    match args.architecture:
        case ArchitectureType.SIMCLR:
            module = SimCLR(
                backbone=backbone_model,
                projection_head=projection_head,
                learning_rate=learning_rate,
                optimizer=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
            )
        case ArchitectureType.MOCO:
            module = MoCo(
                backbone=backbone_model,
                projection_head=projection_head,
                learning_rate=learning_rate,
                optimizer=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                projection_head_output_dim=args.projection_head_output_dim,
                epochs=args.epochs,
            )
        case ArchitectureType.SIMSIAM:
            module = SimSiam(
                backbone=backbone_model,
                projection_head=projection_head,
                prediction_head=prediction_head,
                learning_rate=learning_rate,
                optimizer=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
            )
        case ArchitectureType.BYOL:
            module = BYOL(
                backbone=backbone_model,
                projection_head=projection_head,
                prediction_head=prediction_head,
                learning_rate=learning_rate,
                optimizer=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
            )
        case ArchitectureType.SWAV:
            module = SwaV(
                backbone=backbone_model,
                projection_head=projection_head,
                learning_rate=learning_rate,
                optimizer=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                projection_head_hidden_dim=projection_head_hidden_dim,
                projection_head_output_dim=args.projection_head_output_dim,
                epochs=args.epochs,
            )
        case _:
            raise ValueError(f"Unsupported architecture: {args.architecture}")

    return module


def get_best_model_checkpoint(
    trainer_callbacks: dict[str, Callback],
) -> str | None:
    return (
        trainer_callbacks["ModelCheckpoint"].best_model_path
        if "ModelCheckpoint" in trainer_callbacks
        else None
    )
