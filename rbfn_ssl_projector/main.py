#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import lightning as L

from torch import nn
from pprint import pprint
from torch.utils.data import Dataset
from rbfn_ssl_projector.args import get_args
from rbfn_ssl_projector.utils import (
    seed_rng,
    print_gpu_info,
    create_loggers,
    create_trainer_callbacks,
    get_datamodule,
    get_backbone_model,
    get_heads,
    get_learning_rate,
    get_module,
    get_best_model_checkpoint,
    get_parameter_stats,
)
from logreg.module import LogisticRegressionModule
from vision_datasets.datasets import FeatureDataset
from logreg.datamodule import LogisticRegressionDataModule
from lightning.pytorch.strategies import DDPStrategy

if __name__ == "__main__":
    seed_rng()
    torch.set_float32_matmul_precision("medium")

    ARGS = get_args()
    params = vars(ARGS)
    pprint(params)

    # Display GPU information
    print_gpu_info()

    # ===== Create loggers for SSL architecture =====

    ssl_loggers = create_loggers(
        log_dir=ARGS.log_dir,
        experiment_name=ARGS.experiment_name,
        run_name=ARGS.run_name,
        use_mlflow=ARGS.use_mlflow,
    )

    # ===== Load training and evaluation dataset =====

    ssl_datamodule: L.LightningDataModule = get_datamodule(
        args=ARGS, logger=ssl_loggers.values()
    )

    # ===== Assemble SSL architecture: backbone + projection/prediction head =====

    backbone_model: nn.Module = get_backbone_model(
        backbone=ARGS.backbone, weights=ARGS.weights
    )

    projection_head, prediction_head, head_params = get_heads(
        args=ARGS, backbone_model=backbone_model
    )
    params |= head_params

    print(
        f"Projection head type '{ARGS.projection_head_type}':",
        projection_head,
        list(projection_head.parameters(recurse=True)),
    )

    if prediction_head is not None:
        print(
            f"Prediction head type '{ARGS.projection_head_type}':",
            prediction_head,
            list(prediction_head.parameters(recurse=True)),
        )

    ssl_base_learning_rate, ssl_learning_rate = get_learning_rate(
        architecture=ARGS.architecture,
        base_learning_rate=ARGS.base_learning_rate,
        learning_rate=ARGS.learning_rate,
        batch_size=ARGS.batch_size,
    )
    params |= {
        "base_learning_rate": ssl_base_learning_rate,
        "learning_rate": ssl_learning_rate,
    }

    ssl_module = get_module(
        args=ARGS,
        backbone_model=backbone_model,
        projection_head=projection_head,
        prediction_head=prediction_head,
        projection_head_hidden_dim=params["projection_head_hidden_dim"],
        learning_rate=ssl_learning_rate,
    )
    params |= get_parameter_stats(module=ssl_module, prefix="ssl_")

    ssl_trainer_callbacks = create_trainer_callbacks(
        patience=ARGS.patience, save_top_k=ARGS.save_top_k
    )
    params["ssl_trainer_callbacks"] = list(ssl_trainer_callbacks.keys())

    # ===== SSL Training =====

    ssl_trainer = L.Trainer(
        num_nodes=1,
        devices=ARGS.devices,
        max_epochs=ARGS.epochs,
        callbacks=list(ssl_trainer_callbacks.values()),
        deterministic=True,
        use_distributed_sampler=True,
        sync_batchnorm=True,
        logger=ssl_loggers.values(),
        log_every_n_steps=5,
        precision="bf16-mixed" if ARGS.precision == "bfloat16" else None,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    ssl_trainer.fit(model=ssl_module, datamodule=ssl_datamodule)

    BEST_SSL_MODEL_CHECKPOINT = get_best_model_checkpoint(
        trainer_callbacks=ssl_trainer_callbacks
    )
    params["ssl_model_ckpt"] = BEST_SSL_MODEL_CHECKPOINT

    # ===== Log additional parameters for SSL architecture =====

    params |= ssl_module.hparams
    params = {k: v for k, v in params.items() if v is not None}
    for logger in ssl_loggers.values():
        logger.log_hyperparams(params)

    # ===== Predict with trained SSL backbone on test split for LogReg =====

    ssl_predictor = L.Trainer(
        num_nodes=1,
        devices=ARGS.devices,
        use_distributed_sampler=True,
        deterministic=True,
        sync_batchnorm=True,
        logger=ssl_loggers.values(),
        precision="bf16-mixed" if ARGS.precision == "bfloat16" else None,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    ssl_predictions = ssl_predictor.predict(
        model=ssl_module,
        datamodule=ssl_datamodule,
        return_predictions=True,
        ckpt_path=BEST_SSL_MODEL_CHECKPOINT,
    )

    # ===== Create loggers for LogReg evaluation =====

    logreg_loggers = create_loggers(
        log_dir=ARGS.log_dir,
        experiment_name=ARGS.experiment_eval_name,
        run_name=ARGS.run_name,
        use_mlflow=ARGS.use_mlflow,
    )

    # ===== Create feature datasets for evaluation =====

    logreg_dataset: Dataset = FeatureDataset(
        predictions=ssl_predictions,
        precision=(
            getattr(torch, ARGS.precision) if ARGS.precision is not None else None
        ),
    )

    logreg_datamodule: L.LightningDataModule = LogisticRegressionDataModule(
        dataset=logreg_dataset,
        train_perc=ARGS.eval_train_perc,
        val_perc=ARGS.eval_val_perc,
        test_perc=ARGS.eval_test_perc,
        batch_size=ARGS.eval_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=ARGS.pin_memory,
    )

    # ===== LogReg Training with Feature Dataset =====

    logreg_module: L.LightningModule = LogisticRegressionModule(
        epochs=ARGS.eval_epochs,
        latent_dim=logreg_dataset.latent_dim,
        num_classes=ssl_datamodule.num_classes,
        learning_rate=ARGS.eval_learning_rate,
        weight_decay=ARGS.eval_weight_decay,
    )
    params |= get_parameter_stats(module=logreg_module, prefix="logreg_")

    logreg_trainer_callbacks = create_trainer_callbacks(
        patience=ARGS.eval_patience, save_top_k=ARGS.save_top_k
    )
    params["logreg_trainer_callbacks"] = list(logreg_trainer_callbacks.keys())

    logreg_trainer = L.Trainer(
        num_nodes=1,
        devices=ARGS.devices,
        max_epochs=ARGS.eval_epochs,
        callbacks=list(logreg_trainer_callbacks.values()),
        deterministic=True,
        use_distributed_sampler=True,
        sync_batchnorm=True,
        logger=logreg_loggers.values(),
        log_every_n_steps=1,
        precision="bf16-mixed" if ARGS.precision == "bfloat16" else None,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    logreg_trainer.fit(model=logreg_module, datamodule=logreg_datamodule)

    BEST_LOGREG_MODEL_CHECKPOINT = get_best_model_checkpoint(
        trainer_callbacks=logreg_trainer_callbacks
    )
    params["logreg_model_ckpt"] = BEST_LOGREG_MODEL_CHECKPOINT

    # ===== Log all parameters for LogReg =====

    params |= logreg_module.hparams
    params = {k: v for k, v in params.items() if v is not None}
    for logger in logreg_loggers.values():
        logger.log_hyperparams(params)

    # ===== LogReg Testing =====

    logreg_tester = L.Trainer(
        num_nodes=1,
        devices=ARGS.devices,
        deterministic=True,
        sync_batchnorm=True,
        use_distributed_sampler=False,
        logger=logreg_loggers.values(),
        precision="bf16-mixed" if ARGS.precision == "bfloat16" else None,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    logreg_tester.test(
        model=logreg_module,
        datamodule=logreg_datamodule,
        ckpt_path=BEST_LOGREG_MODEL_CHECKPOINT,
    )
