import gc
import math
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Protocol, cast, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from . import common, dataset
from ..forecasting.models.model_loader import get_models

class ForecastingModel(Protocol):
    """
    Structural type for the forecasting models used in this project.

    Any object that:
      - wraps a torch.nn.Module in `.model`
      - has a `.config` with hyperparameters
      - has a `.scaler` for normalization
      - exposes `.early_stopping`
      - implements `validate(...)`
      - implements `multi_forecasting_hyper_param_tune(...)`
      - optionally provides `_init_model()` and `model_class`
    will satisfy this protocol, regardless of its actual base class.
    """

    model: nn.Module
    config: Any
    scaler: Any
    early_stopping: Any

    def validate(self, *args: Any, **kwargs: Any) -> float:
        ...

    def multi_forecasting_hyper_param_tune(self, data: Any) -> None:
        ...

    # These are used in train_model to construct the underlying nn.Module
    def _init_model(self) -> nn.Module:
        ...

    @property
    def model_class(self) -> Any:
        ...

        
# ---------------------------------------------------------------------------
# Learning rate schedule (from TFB)
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch: int, args) -> None:
    """
    Adjust optimizer learning rate according to args.lradj schedule.

    Supported types:
        - 'type1', 'type2', 'type3'
        - 'constant'
        - 'cosine'
    """
    if args.lradj == "type1":
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.lr}
    elif args.lradj == "cosine":
        total_epochs = getattr(args, "num_epochs", 20)
        lr = args.lr * (0.5 * (1 + math.cos(math.pi * epoch / total_epochs)))
        lr_adjust = {epoch: lr}
    else:
        print(f"[Warning] Unknown lradj '{args.lradj}', keeping current LR.")
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def forecast_fit(model: ForecastingModel, train_dataset, validate_dataset, **kwargs) -> ForecastingModel:
    """
    Train (or fine-tune) the given model on train/validation datasets.

    Handles:
        - optional checkpoint restore
        - attaching transforms
        - DataLoader setup with deterministic seeding
        - optimizer and loss
        - epoch loop (train + validate)
        - early stopping
        - learning rate schedule
    """

        # ensure nn_model is initialized
    if model.model is None:
        raise ValueError("Model not initialized. Call the model initializer first.")
    nn_model = model.model

    # Optional checkpoint restore
    if kwargs.get("use_checkpoint", True) and hasattr(model, "early_stopping"):
        if model.early_stopping.check_point is not None:
            nn_model.load_state_dict(model.early_stopping.check_point)

    # Attach transform to datasets if needed
    if not kwargs.get("already_transformed", False) and getattr(model.config, "norm", False):
        train_dataset.transform = model.scaler.transform
        validate_dataset.transform = model.scaler.transform

    config = model.config

    num_workers = getattr(config, "num_workers", 0) or 0

    # Optional deterministic seeding for workers and shuffling
    base_seed = kwargs.get("seed", None)

    def _worker_init_fn(worker_id: int) -> None:
        """
        Worker init function to set different random seeds per worker.
        """
        if base_seed is None:
            return
        seed = base_seed + worker_id
        import random as _random
        import numpy as _np
        import torch as _torch
        _random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)

    # DataLoader setup
    data_generator = (
        torch.Generator().manual_seed(int(base_seed))
        if base_seed is not None else None
    )

    # Use pin_memory if CUDA is available
    use_pin_memory = torch.cuda.is_available()

    train_data_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=_worker_init_fn,
        generator=data_generator,
        pin_memory=use_pin_memory,
    )
    valid_data_loader = DataLoader(
        validate_dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=_worker_init_fn,
        generator=data_generator,
        pin_memory=use_pin_memory,
    )

    # Loss selection
    if config.loss == "MSE":
        criterion = nn.MSELoss()
    elif config.loss == "MAE":
        criterion = nn.L1Loss()
    else:
        criterion = nn.HuberLoss(delta=0.5)

    # Optimizer selection
    opt_name = getattr(config, "optimizer", "adam").lower()
    weight_decay = getattr(config, "weight_decay", 0.0)

    if opt_name == "adam":
        optimizer = optim.Adam(nn_model.parameters(), lr=config.lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(nn_model.parameters(), lr=config.lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            nn_model.parameters(), lr=config.lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup early stopping
    model.early_stopping = common.EarlyStopping(patience=config.patience)

    # Move model to device
    nn_model.to(device)

    # AMP: mixed precision
    # Logic: AMP is activated, if:
    # 1. A GPU is available (cuda)
    # 2. And the Config allows it

    print("cuda is available: " + str(torch.cuda.is_available()))

    config_use_amp = getattr(config, "use_amp", True) # default to True
    use_amp = (device.type == "cuda") and config_use_amp

    if use_amp:
        print("Using AMP (mixed precision) for training.")
    else:
        print("AMP disabled or no CUDA device detected.")

    scaler = GradScaler(enabled=use_amp)

    total_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    for epoch in range(config.num_epochs):
        train_loss_sum = 0.0
        num_batches = 0
        last_grad_norm: Optional[float] = None

        # Randomize global time offset for datasets (shifts time marks)
        if kwargs.get("random_time_offset_per_epoch", False):
            time_offset = int(np.random.randint(2**14, size=1)[0])
            train_data_loader.dataset.time_offset = time_offset  # type: ignore
            valid_data_loader.dataset.time_offset = time_offset  # type: ignore

        # Set model to train mode and iterate batches
        nn_model.train()
        for i, (input, target, input_mark, target_mark) in enumerate(train_data_loader):
            optimizer.zero_grad()

            if torch.isnan(input).any():
                print(f"CRITICAL: NaNs found in INPUT data at batch {i}!")
                # Zeige welche Spalte kaputt ist
                for feat_idx in range(input.shape[-1]):
                    if torch.isnan(input[:, :, feat_idx]).any():
                        print(f" -> Feature Index {feat_idx} contains NaNs (likely variance=0)")
                raise ValueError("NaNs in input data.")

            input = input.to(device)
            target = target.to(device)
            input_mark = input_mark.to(device)
            target_mark = target_mark.to(device)


            loss_importance: Optional[torch.Tensor] = None  # reset every batch

            # --- Forward pass (AMP-aware) ---
            with torch.cuda.amp.autocast(enabled=use_amp):
                if config.decoder_input_required:
                    # Create a zero matrix shaped like the forecast horizon
                    dec_input = torch.zeros_like(
                        target[:, -config.horizon :, :]
                    ).float()

                    # Build decoder input: [past labels] + [zero horizon placeholder]
                    dec_input = (
                        torch.cat(
                            [target[:, : config.label_len, :], dec_input], dim=1
                        )
                        .float()
                        .to(device)
                    )
                    # Forward pass with decoder input
                    output = nn_model(input, input_mark, dec_input, target_mark)
                elif config.has_loss_importance:
                    # Forward pass with loss importance
                    output, loss_importance = nn_model(input)
                else:
                    # Standard forward pass
                    output = nn_model(input)

                # Focus loss on forecast horizon
                target_slice = target[:, -config.horizon :, :]
                output_slice = output[:, -config.horizon :, :]
                loss = criterion(output_slice, target_slice)

                if getattr(config, "has_loss_importance", False):
                    loss = loss + loss_importance

            # --- Backward pass and Optimizer step ---
            if use_amp:
                scaler.scale(loss).backward()

                # Gradient norm (unscaled) for logging
                try:
                    scaler.unscale_(optimizer)  # make grads real for norm & clipping
                    total_norm_sq = 0.0
                    for p in nn_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm_sq += param_norm * param_norm
                    last_grad_norm = total_norm_sq ** 0.5
                except Exception:
                    last_grad_norm = None

                if hasattr(config, "grad_clip") and config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        nn_model.parameters(), config.grad_clip
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                # Gradient norm for logging
                try:
                    total_norm_sq = 0.0
                    for p in nn_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm_sq += param_norm * param_norm
                    last_grad_norm = total_norm_sq ** 0.5
                except Exception:
                    last_grad_norm = None

                if hasattr(config, "grad_clip") and config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        nn_model.parameters(), config.grad_clip
                    )

                optimizer.step()

            train_loss_sum += float(loss.item())
            num_batches += 1

        # Validation
        series_dim = getattr(config, "c_out", None)
        if series_dim is None:
            # Fallback: infer from first batch: target shape [B, T, D]
            first_batch = next(iter(valid_data_loader))
            series_dim = first_batch[1].shape[-1]

        try:
            valid_loss = model.validate(valid_data_loader, series_dim, criterion)
        except TypeError:
            # For legacy models: validate(valid_data_loader, criterion)
            valid_loss = model.validate(valid_data_loader, criterion)  # type: ignore

        # Optional training progress callback
        ProgressCallback = Callable[
            [float, float, ForecastingModel, int, Dict[str, Any]],
            None
        ]

        raw_callback = kwargs.get("training_progress_callback", None)
        callback_fcn: Optional[ProgressCallback] = (
            raw_callback if callable(raw_callback) else None
        )

        if callback_fcn:
            avg_train_loss = train_loss_sum / max(1, num_batches)
            current_lr = optimizer.param_groups[0].get("lr", None)
            callback_fcn(
                valid_loss,
                model.early_stopping.val_loss_min,
                model,
                epoch,
                {
                    "train_loss": avg_train_loss,
                    "lr": current_lr,
                    "grad_norm": last_grad_norm,
                },
            )

        if model.early_stopping(valid_loss, model.model):
            break

        adjust_learning_rate(optimizer, epoch + 1, config)

    return model


# ---------------------------------------------------------------------------
# Orchestration: end-to-end training
# ---------------------------------------------------------------------------

def train_model(
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    High-level training entry point.

    Steps:
        1. Apply global seed from evaluation_config (if present)
        2. Build model from model_config (using TFB factory)
        3. Load & split datasets (normal + overrides)
        4. Scale data and create CustomDatasetWithOverrides instances
        5. Run forecast_fit() training loop
        6. Return validation loss, training time, and model size (in M parameters)
    """

    # Apply seed from evaluation config; prefer strategy_args.seed
    seed = None
    if isinstance(evaluation_config, dict):
        strat_args = evaluation_config.get("strategy_args", {})
        seed = strat_args.get("seed", evaluation_config.get("seed"))
    if seed is not None:
        common.set_fixed_seed(seed)

    # Instantiate the model via TFB factory
    model_factory = get_models(model_config)[0]
    raw_model = model_factory()
    model: ForecastingModel = cast(ForecastingModel, raw_model)

    # Push config params from model_config into model.config
    for key, val in model_config["models"][0].items():
        setattr(model.config, key, val)

    # Debug: print config contents
    for key, val in model.config.__dict__.items():
        print(key, val)

    # Resolve dataset paths with env/kwargs defaults
    data_path, data_path_overrides = common.resolve_data_paths(
        kwargs.get("data_path"), kwargs.get("data_path_overrides")
    )

    data = common.load_csv(data_path)
    data_w_overrides = common.load_csv(data_path_overrides)

    # Task name for TFB internals
    setattr(model.config, "task_name", "short_term_forecast")

    # --- following code is extraced from TFB's multi_forecasting_hyper_param_tune
    # TODO check if this step is necessary
    column_num = data.shape[1]
    model.config.enc_in = column_num
    model.config.dec_in = column_num
    model.config.c_out = column_num

    if model.model_name == "MICN":
        setattr(model.config, "label_len", model.config.seq_len)
    else:
        setattr(model.config, "label_len", model.config.seq_len // 2)
    # ---

    # Initialize underlying PyTorch model
    if hasattr(model, "_init_model"):
        model.model = model._init_model()
    else:
        model.model = model.model_class(model.config)

    lookback = model_config["models"][0]["model_hyper_params"]["seq_len"]
    horizon = evaluation_config["strategy_args"]["horizon"]
    lookahead = horizon

    # sliding window stride (in samples)
    strategy_args = evaluation_config.get("strategy_args", {})
    sliding_stride = int(strategy_args.get("sliding_stride", 1))

    # Ensure TimesNet / TSL models see the right pred_len
    # (safe for other models; extra attr is just ignored if unused)
    setattr(model.config, "pred_len", horizon)

    train_ratio_in_tv = evaluation_config["strategy_args"]["train_ratio_in_tv"]
    tv_ratio = evaluation_config["strategy_args"]["tv_ratio"]  # train+val fraction

    # Train/val split (ignore test here)
    train_data, valid_data, _ = common.split_data(data, tv_ratio, train_ratio_in_tv)
    train_data_w_overrides, valid_data_w_overrides, _ = common.split_data(
        data_w_overrides, tv_ratio, train_ratio_in_tv
    )

    # Fit scaler on both normal and override train data
    model.scaler.partial_fit(train_data.values)
    model.scaler.partial_fit(train_data_w_overrides.values)

    # Scale copies for training and validation
    train_data = common.scaled_dataframe_copy(train_data, model.scaler)
    valid_data = common.scaled_dataframe_copy(valid_data, model.scaler)
    train_data_w_overrides = common.scaled_dataframe_copy(
        train_data_w_overrides, model.scaler
    )
    valid_data_w_overrides = common.scaled_dataframe_copy(
        valid_data_w_overrides, model.scaler
    )

    gc.collect()

    generate_temporal_features=getattr(model.config, "generate_temporal_features", False)
    freq = getattr(model.config, "freq")

    # Build datasets
    train_dataset = dataset.CustomDatasetWithOverrides(
        train_data,
        train_data_w_overrides,
        lookback,
        lookahead,
        model.config.label_len,
        model.config.input_sampling,
        stride_samples=sliding_stride,
        generate_temporal_features=generate_temporal_features,
        freq=freq,

    )
    validate_dataset = dataset.CustomDatasetWithOverrides(
        valid_data,
        valid_data_w_overrides,
        lookback,
        lookahead,
        model.config.label_len,
        model.config.input_sampling,
        stride_samples=sliding_stride,
        generate_temporal_features=generate_temporal_features,
        freq=freq,
    )

    print(f"Train Dataset Size: {len(train_dataset)} samples")
    print(f"Valid Dataset Size: {len(validate_dataset)} samples")
    
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Check seq_len/horizon vs. data length.")

    start_time = time.time()
    forecast_fit(
        model,
        train_dataset,
        validate_dataset,
        use_checkpoint=False,
        random_time_offset_per_epoch=True,
        already_transformed=True,
        training_progress_callback=kwargs.get("training_progress_callback", None),
        seed=seed,
    )
    elapsed = time.time() - start_time

    try:
        params_millions = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        params_millions = None

    result = {
        "val_loss": getattr(model.early_stopping, "val_loss_min", float("inf")),
        "training_time": elapsed,
        "model_params_millions": params_millions,
    }
    return result