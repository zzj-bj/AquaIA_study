import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import get_scheduler

from dataloading.datasets import JpgDALIDataset, DALIDetectionDataLoader, JpgDetectionDataset, parse_batch, detection_collate_fn
from detection.dino.dino_detector import DINODetector
from detection.dino.loss import SetCriterion
from detection.metric import compute_metrics, print_metrics, update_metric_dict
from detection.dino.utils.matcher import HungarianMatcher
from detection.utils.config_utils import save_resolved_config
from detection.utils.plot_utils import plot_metrics, save_sample_predictions
from detection.dino.predict import predict, normalize_imgsz
from detection.utils.import_utils import DALI_AVAILABLE
from detection.logging import TrainingLogger, CheckpointManager, register_run, update_run_status


def get_datasets(
    data_yaml_path,
    batch_size,
    device,
    img_size=640,
    loader="jpg",
):
    """Z: Create the training and validation datasets, return train_dataset, val_dataset, num_classes.
    It chooses different dataset implementations depending on whether the current environment supports DALI."""
    # TODO : currently GPU only because of DALI, but should be possible to support CPU-only training)
    # Compute random split for train and eval set
    if DALI_AVAILABLE:
        train_dataset = JpgDALIDataset(
            dataset_root=data_yaml_path,
            data_split="train",
            img_size=img_size,
            batch_size=batch_size,
            device=device,
        )
        val_dataset = JpgDALIDataset(
            dataset_root=data_yaml_path,
            data_split="val",
            img_size=img_size,
            batch_size=batch_size,
            device=device,
        )
    else:
        train_dataset = JpgDetectionDataset(
            dataset_root=data_yaml_path,
            data_split="train",
            img_size=img_size,
            device=device,
        )
        val_dataset = JpgDetectionDataset(
            dataset_root=data_yaml_path,
            data_split="val",
            img_size=img_size,
            device=device,
        )
    num_classes = train_dataset.num_classes
    return train_dataset, val_dataset, num_classes


def build_scheduler(training_config, optimizer):
    """Z: Create lr scheduler, warmup + cosine decay + min lr constraint."""
    if not training_config.get("cos_lr", False):
        return None
    warmup_ratio = float(training_config.get("warmup_ratio", 0.0))
    # Z: epoch level scheduler, not batch level
    warmup_steps = int(training_config["epochs"] * warmup_ratio)
    if warmup_ratio > 0.0:
        warmup_steps = max(warmup_steps, 1)
    scheduler = get_scheduler(
        name="cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_config["epochs"],
        # Z: min lr for cosine scheduler = lrf * lr0
        scheduler_specific_kwargs={"min_lr_rate": training_config.get("lrf", 0.01)},
    )
    return scheduler


def train_dino(config, resume_dir=None):
    """Z: Read training parameters from the configuration, create the dataset and model,
    execute the training and validation loop, and save checkpoints, logs, metrics, and prediction results."""
    training_config = config["training"]
    output_config = config["output"]
    log_config = config.get("logging", {})

    configured_device = training_config.get("device", "auto")
    if configured_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = configured_device

    # Z: use amp if device is cuda
    use_amp = device == "cuda"

    imgsz = normalize_imgsz(config, "training")
    train_set, val_set, num_classes = get_datasets(
        config["data"]["dataset_yaml"],
        training_config["batch"],
        device=device,
        img_size=imgsz,
        loader=config["data"].get("loader", "jpg"),
    )

    # === Setup dataloaders ===
    if DALI_AVAILABLE:
        train_dataloader = DALIDetectionDataLoader(train_set, device="gpu")
        val_dataloader = DALIDetectionDataLoader(val_set, device="gpu")
    else:
        # Z: pin_memory=True + non_blocking=True accelerates data transfer from CPU to GPU
        train_dataloader = DataLoader(train_set, batch_size=training_config["batch"], shuffle=True, num_workers=3, pin_memory=True, collate_fn=detection_collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=training_config["batch"], shuffle=False, num_workers=3, collate_fn=detection_collate_fn)

    # === Run directory ===
    if resume_dir:
        run_dir = str(resume_dir)
        run_id = Path(run_dir).name
    else:
        # Z: "20260705_143208"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Z: "runs/20260705_143208"
        run_dir = os.path.join(output_config.get("project", "runs"), run_id)
    os.makedirs(run_dir, exist_ok=True)
    # Z: "runs/20260705_143208/weights"
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # === Logging & checkpointing ===
    logger = TrainingLogger(run_dir=run_dir, run_id=run_id, config=config, resume=bool(resume_dir))
    checkpoint_mgr = CheckpointManager(
        run_dir=run_dir,
        save_period=log_config.get("save_period", 0),
    )
    # Z: if new training run
    if not resume_dir:
        register_run(config=config, run_id=run_id, run_dir=run_dir, pid=os.getpid())
    logger.log_device(
        device=device,
        use_amp=use_amp,
        dataset_info=f"{config['data']['dataset_yaml']} (train={len(train_set)} val={len(val_set)})",
    )

    # === DINO model ===
    backbone_config = config.get("model", {})
    backbone_family = str(backbone_config.get("family", "")).lower()
    backbone_size = str(backbone_config.get("size", "").lower())

    model = DINODetector(
        backbone_id=backbone_family + "_" + backbone_size,
        img_size=imgsz,
        device=device,
        num_classes=num_classes,
    ).to(device)

    matcher = HungarianMatcher(
        cost_class=training_config["cost_class"],
        cost_bbox=training_config["cost_bbox"],
        cost_giou=training_config["cost_giou"],
        cost_bbox_type=training_config["cost_bbox_type"],
    )
    # Z: SetCriterion computes various losses
    # Z: then training loop combines through this dictionary into the total loss.
    loss_weight_dict = {
        "loss_ce": training_config["cls"],
        "loss_bbox": training_config["box"],
        "loss_giou": training_config["giou"],
    }
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
    ).to(device)

    # Z: get optimizer class then initialize it
    optimizer_cls = getattr(torch.optim, training_config.get("optimizer", "AdamW"))
    optimizer = optimizer_cls(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config["lr0"],
        weight_decay=training_config["weight_decay"],
    )

    scheduler = build_scheduler(training_config, optimizer)

    # Z: create a GradScaler to prevent gradient underflow when using AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # Z: step 1 scale the loss then backpropagate the scaled loss producing scaled gradients
    # Z:    scaler.scale(total_loss).backward()
    # Z: step 2 unscale gradients then check whether they contain inf or NaN
    # Z: execute optimizer.step() if gradients are finite; otherwise skip this parameter update
    # Z:    scaler.step(optimizer)
    # Z: step 3 dynamically adjust the scale factor
    # Z:    scaler.update()
    best_validation_loss = float("inf")
    metrics_history = []
    start_epoch = 0

    # === Load checkpoint if resuming (before compile) ===
    if resume_dir:
        from detection.checkpoint import load_training_state_checkpoint

        last_weights = os.path.join(run_dir, "weights", "last.pt")
        last_state = os.path.join(run_dir, "last_training_state.pt")

        if os.path.exists(last_weights):
            # Z: load the checkpoint from disk and move tensors to device
            ckpt = torch.load(last_weights, map_location=device)
            # Z: find the model that actually need to receive weights
            # Z: model compilation may add some additional attributes, take original model
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            # Z: load the model state dict from the checkpoint into the model
            base_model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"[RESUME] Loaded model weights from {last_weights}")

        if os.path.exists(last_state):
            state = load_training_state_checkpoint(last_state, device=device)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            # Z: resume number of achived epochs
            start_epoch = state["epoch"]
            logger.info(f"[RESUME] Resuming from epoch {start_epoch + 1}/{training_config['epochs']}")

        meta_path = Path(run_dir) / "run_meta.json"
        if meta_path.exists():
            # Z: read the JSON text and convert it to a Python dictionary
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            best_validation_loss = float(meta.get("best_val_loss") or "inf")

    model.train()
    criterion.train()
    if training_config.get("compile", False):
        # Z: torch.compile() attempts to optimize the model's forward computation graph
        # Z: to make training or inference faster
        model = torch.compile(model)

    # === Training loop ===
    try:
        # Z: START for epoch loop
        for epoch in range(start_epoch, training_config["epochs"]):
            metric_dict = {"train": {}, "val": {}}
            logger.info(f"========== Epoch {epoch + 1}/{training_config['epochs']} ==========")

            # Z: START for each train val dataloader -> for each epoch first train then val
            for loader in [train_dataloader, val_dataloader]:
                split = loader.dataset.data_split
                training = split == "train"

                metric_dict[split]["loss"] = 0.0
                # Z: batch level loader progress bar, hence train val batch level progress bar
                progress = tqdm.tqdm(loader, desc="- Training   " if training else "- Validation ", unit="batch")

                if not training:
                    model.eval()
                else:
                    model.train()

                with torch.set_grad_enabled(training):
                    # Z: START for batch loop
                    for batch_idx, batch in enumerate(progress):
                        # Z: Non-DALI: batch = { "images": Tensor[B, 3, H, W], "inputs": Tensor[B, 3, H, W],
                        # Z: "targets_idx": list[int], "img_paths": list[str], }
                        # Z: DALI: batch = { "inputs": Tensor[B, 3, H, W], "targets_idx": Tensor or DALI output }
                        # Z: targets =[ {"labels": ..., "boxes": ...}, {"labels": ..., "boxes": ...}, ...]
                        targets = loader.dataset.get_targets(batch)
                        images, _ = parse_batch(batch)

                        if not DALI_AVAILABLE:
                            images = images.to(device, non_blocking=True)

                        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                            # Z: outputs = { "pred_logits": tensor(...), "pred_boxes": tensor(...), }
                            outputs = model(images)
                            # Z: loss_dict = { "loss_ce": ..., "class_error": ..., "loss_bbox": ...,
                            # Z: "loss_giou": ..., "cardinality_error": ... } batch level
                            loss_dict = criterion(outputs, targets)
                        total_loss = sum(loss_dict[key] * loss_weight_dict[key] for key in loss_dict if key in loss_weight_dict)

                        if training:
                            # Z: set_to_none=True set gradients to None, reducing memory usage
                            optimizer.zero_grad(set_to_none=True)
                            # Z: scale the loss then backpropagate the scaled loss producing scaled gradients
                            scaler.scale(total_loss).backward()
                            # Z: unscale gradients then check whether they contain inf or NaN
                            # Z: execute optimizer.step() if gradients are finite; otherwise skip this parameter update
                            scaler.step(optimizer)
                            # Z: dynamically adjust the scale factor
                            scaler.update()

                        batch_loss = float(total_loss.item())
                        # Z: show the current batch loss values in the progress bar, formatted to 4 decimal places
                        progress.set_postfix(
                            **{key: f"{float(value.item()):.4f}" for key, value in loss_dict.items() if key in loss_weight_dict},
                        )
                        # Z: loss_dict, batch_loss are batch level, metric_dict is epoch level, progress.total = nb batches
                        update_metric_dict(metric_dict, loss_dict, batch_loss, loader.dataset.data_split, progress.total)

                        # Heartbeat — updated every N batches
                        # Z: +1 because epoch and batch_idx are 0-indexed but we want to log 1-indexed values
                        logger.heartbeat(epoch + 1, batch_idx + 1, progress.total)

                    # Z: END for batch loop

                    if training and scheduler is not None:
                        scheduler.step()

            # Z: END for each train val dataloader

            metric_dict["epoch"] = epoch + 1
            # Z: save epoch-level metrics to history and print them
            metrics_history.append(metric_dict)
            print_metrics(metric_dict)

            # Epoch-level logging
            # Z: read first param group lr
            lr = optimizer.param_groups[0]["lr"]
            logger.log_epoch(epoch + 1, training_config["epochs"], metric_dict, lr)

            # Checkpoint: best and periodic last
            validation_loss = metric_dict["val"]["loss"]
            is_best = validation_loss <= best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                logger.log_best(epoch + 1, validation_loss)
            checkpoint_mgr.step(epoch + 1, model, optimizer, scaler, scheduler, is_best=is_best)

        # Z: END for epoch loop

        # === Training ended normally ===
        checkpoint_mgr.save_final(training_config["epochs"], model, optimizer, scaler, scheduler)
        logger.finish()
        update_run_status(config, run_id, "done")

    # Z: like Ctrl + C
    except KeyboardInterrupt:
        logger.interrupted()
        update_run_status(config, run_id, "interrupted")
        # Save partial metrics so the run isn't a total loss
        if metrics_history:
            np.save(os.path.join(run_dir, "metrics.npy"), metrics_history, allow_pickle=True)
        # Z: still raise exception so program exits
        raise

    except Exception as exc:
        logger.crash(str(exc))
        update_run_status(config, run_id, "error")
        if metrics_history:
            np.save(os.path.join(run_dir, "metrics.npy"), metrics_history, allow_pickle=True)
        raise

    # === Post-training: metrics, config, eval ===
    # Z: metrics_history = [ { "train": {"loss": ..., "loss_ce": ..., "loss_bbox": ..., "loss_giou": ..., "class_error":..., "cardinality_error":... },
    # Z: "val": {"loss": ..., "loss_ce": ..., "loss_bbox": ..., "loss_giou": ..., "class_error":..., "cardinality_error":... }, "epoch": 1 },... ]
    np.save(os.path.join(run_dir, "metrics.npy"), metrics_history, allow_pickle=True)
    plot_metrics(run_dir)

    save_resolved_config(
        path=os.path.join(run_dir, "resolved_config.yaml"),
        config=config,
        device=device,
        use_amp=use_amp,
        run_dir=run_dir,
    )

    # Z: load the checkpoint from disk and move tensors to device
    best_checkpoint = torch.load(os.path.join(weights_dir, "best.pt"), map_location=device)
    # Z: find the model that actually need to receive weights
    best_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Z: load the model state dict from the checkpoint into the model
    best_model.load_state_dict(best_checkpoint["model_state_dict"])

    best_model.eval()
    # Z: compute metrics on train and val sets using the best model
    metrics = compute_metrics(
        model=best_model,
        dataloaders=[train_dataloader, val_dataloader],
        predict_fn=predict,
        device=device,
        # Z: prediction conf !Warining! no "conf_thresh" in train_config.yaml
        conf_thresh=training_config.get("conf_thresh", 0.05),
    )
    logger.info(str(metrics))
    np.save(os.path.join(run_dir, "best_metric.npy"), metrics, allow_pickle=True)

    save_sample_predictions(
        model=best_model,
        subset=val_set,
        predict_fn=predict,
        output_dir=Path(run_dir) / "eval_predictions",
        conf=0.3,
        seed=42,
        device=device,
    )
    save_sample_predictions(
        model=best_model,
        subset=train_set,
        predict_fn=predict,
        output_dir=Path(run_dir) / "train_predictions",
        conf=training_config.get("conf", 0.3),
        seed=42,
        device=device,
    )

    return best_model
