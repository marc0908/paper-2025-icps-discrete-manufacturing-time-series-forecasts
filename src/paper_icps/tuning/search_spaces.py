"""
Improved Search Spaces with Better Parameter Distributions and Ranges
"""
from ray import tune
from paper_icps.core import config
import math

def timexer_searchspace():
    """Enhanced TimeXer search space with better parameter distributions"""
    search_space = {
        # Core architecture parameters
        "batch_size": tune.choice([16, 32, 64, 128, 256]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "d_ff": tune.choice([128, 256, 512, 1024, 2048, 4096, 8192]),  # Expanded range
        "e_layers": tune.choice([1, 2, 3, 4, 5]),  # Added 5 layers
        "n_heads": tune.choice([4, 8, 12, 16]),  # Powers of 2 for efficiency
        
        # New: optimization strategy
        "optimizer": tune.choice(["adam", "adamw", "sgd"]),
        "weight_decay": tune.loguniform(1e-7, 1e-2),

        # New: activation & normalization
        "activation": tune.choice(["gelu", "relu", "silu", "elu"]),
        "normalization": tune.choice(["layernorm", "batchnorm", "rmsnorm"]),

        # Regularization - using log-uniform for better exploration
        "dropout": tune.loguniform(1e-6, 0.5),  # Log-uniform for better exploration
        
        # Learning parameters
        "lr": tune.loguniform(1e-6, 5e-2),
        "lradj": tune.choice(["type1", "type2", "cosine", "none"]),  # More LR schedules
        
        # Model-specific parameters
        "patch_len": tune.choice([8, 16, 32, 64, 100, 200]),  # More granular choices
        "features": "M",  # Fixed for multivariate
        "use_norm": tune.choice([True, False]),  # Let optimization decide
        "factor": tune.randint(1, 10),  # Reduced range for stability
        
        # Training parameters
        "horizon": 400,  # Fixed based on your domain
        "seq_len": 1600,  # Fixed based on your domain
        "loss": "MAE",  # Could be made tunable: tune.choice(["MAE", "MSE"])
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),  # Tunable patience
        "moving_avg": tune.choice([1, 3, 5, 7]),  # Added 7
        
        # Advanced parameters
        "grad_clip": tune.uniform(0.5, 4.0),  # Gradient clipping
        "label_smoothing": tune.uniform(0.0, 0.1),  # Label smoothing for robustness
    }
    return search_space

def fedformer_searchspace():
    """
    Hyperparameter search space for FEDformer (Fourier Enhanced Decomposed Transformer),
    based strictly on the paper FEDformer (Zhou et al., 2022).
    """

    return {
        # ===== Core architecture =====
        # Encoder/Decoder layers (paper uses N=2, M=1 or similar small depth)
        "e_layers": tune.choice([1, 2, 3]),
        "d_layers": tune.choice([1, 2]),

        # Hidden dimension D — Table figs suggest 64–512 depending on dataset
        "d_model": tune.choice([64, 128, 256, 512]),

        # FFN width — multiples of d_model shown in architecture
        "d_ff": tune.choice([256, 512, 1024, 2048]),

        # Attention heads — typical 4–8 for these model sizes
        "n_heads": tune.choice([4, 8]),

        # ===== Frequency domain parameters =====
        # Number of Fourier/Wavelet modes (Section 4.3 + Figure 6) — default M=64
        "modes": tune.choice([16, 32, 64, 96, 128]),

        # Fourier or Wavelet block
        # FEB-f / FEB-w and FEA-f / FEA-w as per Section 3.2 / 3.3
        "domain": tune.choice(["fourier", "wavelet"]),

        # Wavelet levels L (Appendix D)
        "wavelet_levels": tune.choice([2, 3, 4]),

        # Activation for frequency attention (Section 3.2 FEA-f)
        "fea_activation": tune.choice(["softmax", "tanh"]),

        # ===== Mixture-of-Experts Seasonal-Trend Decomposition =====
        # Based on MOEDecomp (Section 3.4)
        "num_experts": tune.choice([3, 4, 5, 6]),

        # kernel sizes from Appendix F.5: [7, 12, 14, 24, 48]
        "expert_kernel_sizes": tune.choice([
            [7, 12, 14, 24, 48],
            [7, 14, 24],
            [12, 24, 48],
        ]),

        # ===== Learning & Training =====
        "batch_size": tune.choice([16, 32, 64]),

        # Paper uses Adam with lr=1e-4 (Appendix F.2) — allow small variation
        "lr": tune.loguniform(3e-5, 3e-4),

        # For stability, match TimesNet defaults
        "weight_decay": tune.loguniform(1e-6, 1e-3),

        # Dropout — paper uses small dropout (0.05–0.1)
        "dropout": tune.uniform(0.0, 0.2),

        # ===== Fixed for your pipeline =====
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MSE",
        "norm": True,

        # ===== Stability parameters =====
        "grad_clip": tune.uniform(0.5, 2.0),
        "patience": tune.choice([5, 10, 15]),
        "moving_avg": tune.choice([1, 3, 5]),
    }

def autoformer_searchspace(num_vars: int | None = None):
    """
    Autoformer search space for long-term multivariate forecasting.

    Based on:
    - d_model = 512, 2 encoder layers, 1 decoder layer in the paper
    - moving_avg window k = 25
    - Adam, lr ~ 1e-4, L2 loss, batch_size = 32, early stop ~10 epochs  [oai_citation:0‡Autoformer.pdf](sediment://file_000000005f3871f4bd01b92c9dbf9c2f)
    """

    # If we know channel count, bias d_model around a power of 2 close to C
    if num_vars is not None and num_vars > 0:
        base = 2 ** math.ceil(math.log2(num_vars))
    else:
        base = 64

    d_model_candidates = sorted(
        {max(32, min(base * m, 512)) for m in [1, 2, 4, 8]}
    )
    d_model_candidates = [d for d in d_model_candidates if 32 <= d <= 512]

    search_space = {
        # === Capacity / architecture ===
        "d_model": tune.choice(d_model_candidates or [128, 256, 512]),
        # Paper: 2 encoder, 1 decoder layer; we keep it shallow
        "e_layers": tune.choice([2, 3]),
        "d_layers": tune.choice([1, 2]),
        # FFN width – modest multiples of d_model
        "d_ff": tune.choice([256, 512, 1024]),
        "n_heads": tune.choice([4, 8]),

        # decomposition (moving average) window – around k=25 in the paper  [oai_citation:1‡Autoformer.pdf](sediment://file_000000005f3871f4bd01b92c9dbf9c2f)
        "moving_avg": tune.choice([7, 13, 25, 49]),

        # Auto-Correlation hyper-parameter c (Top-k periods: k = c * log L)  [oai_citation:2‡Autoformer.pdf](sediment://file_000000005f3871f4bd01b92c9dbf9c2f)
        "c": tune.randint(1, 4),   # {1, 2, 3}

        # === Optimization & regularization ===
        # Centered around 1e-4 as in the paper, with a modest range  [oai_citation:3‡Autoformer.pdf](sediment://file_000000005f3871f4bd01b92c9dbf9c2f)
        "lr": tune.loguniform(3e-5, 3e-4),
        "dropout": tune.uniform(0.05, 0.25),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),

        # === Fixed / training control ===
        "loss": "MSE",            # L2 in the paper
        "horizon": 400,
        "seq_len": 1600,
        "norm": True,
        "num_epochs": config.max_epochs,
        # paper early-stops within ~10 epochs; keep similar scale  [oai_citation:4‡Autoformer.pdf](sediment://file_000000005f3871f4bd01b92c9dbf9c2f)
        "patience": tune.choice([5, 10, 15]),
        "grad_clip": tune.uniform(0.5, 2.0),
    }

    return search_space

def timesnet_searchspace(num_vars: int | None = None):
    """
    TimesNet search space for long-term multivariate forecasting.

    Follows the paper:
    - shallow depth (2–3 TimesBlocks)
    - moderate d_model (32–512)
    - small top_k (3–5)
    - lr around 1e-4, small dropout, Adam, MSE
    """
    # If we know the channel count, use the paper's rule d_model ≈ 2^{ceil(log2 C)}
    if num_vars is not None and num_vars > 0:
        base = 2 ** math.ceil(math.log2(num_vars))
    else:
        # Fallback prior if we don't know C
        base = 64

    # Candidate d_model values around the "base", clamped to [32, 512]
    d_model_candidates = sorted(
        {max(32, min(base * m, 512)) for m in [1, 2, 4, 8]}
    )
    # Make sure we only keep sane values
    d_model_candidates = [d for d in d_model_candidates if 32 <= d <= 512]

    search_space = {
        # === Capacity ===
        "d_model": tune.choice(d_model_candidates or [64, 128, 256, 512]),
        # TimesNet uses 2 layers for most forecasting tasks; try 2–3 only.
        "e_layers": tune.choice([2, 3]),
        # top-k frequencies in FFT; paper shows low sensitivity for k ≤ 5
        "top_k": tune.choice([3, 5]),
        # FFN / conv width (simple multiples of d_model)
        "d_ff": tune.choice([128, 256, 512, 1024]),

        # === Optimization & regularization ===
        # Centered around 1e-4 as in the paper, with ~×3 range
        "lr": tune.loguniform(3e-5, 3e-4),
        "dropout": tune.uniform(0.05, 0.20),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),

        # === Fixed / training control ===
        "loss": "MSE",
        "horizon": 400,        # adjust if your eval_config uses something else
        "seq_len": 1600,       # same as other models in your setup
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([5, 10, 15]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        "grad_clip": tune.uniform(0.5, 2.0),
    }

    return search_space

def duet_searchspace():
    """Enhanced DUET search space"""
    search_space = {
        "CI": 1,  # Channel independent - fixed
        "batch_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([128, 256, 512, 1024]),
        "d_model": tune.choice([128, 256, 512, 768]),  # Added 768
        "e_layers": tune.choice([1, 2, 3, 4]),
        
        # Better regularization
        "dropout": tune.loguniform(1e-4, 0.25),
        "fc_dropout": tune.loguniform(1e-5, 0.2),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Learning parameters
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        
        # Architecture parameters
        "factor": tune.randint(1, 15),  # Slightly reduced
        "n_heads": tune.choice([4, 8, 12, 16, 20]),  # More structured choices
        "num_experts": tune.randint(2, 16),  # Reduced max for stability
        "k": tune.choice([1, 2, 3, 4]),  # Added 4
        
        # Domain-specific
        "horizon": 400,
        "seq_len": 1600,
        "patch_len": tune.choice([8, 16, 32, 50, 100, 200, 400]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        
        # Fixed parameters
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
        
        # Additional parameters
        "grad_clip": tune.uniform(0.5, 2.0),
    }
    return search_space

def nonstationary_transformer_searchspace(num_vars: int | None = None):
    """
    Search space for Non-stationary Transformer (vanilla Transformer backbone
    wrapped with Series Stationarization + De-stationary Attention).

    - Centered around the paper's config: d_model=512, e_layers=2, lr=1e-4.
    - Slight flexibility in depth/width.
    """

    # heuristic for d_model based on channel count, similar to TimesNet
    if num_vars is not None and num_vars > 0:
        base = 2 ** math.ceil(math.log2(num_vars))
    else:
        base = 64

    d_model_candidates = sorted(
        {max(64, min(base * m, 512)) for m in [1, 2, 4, 8]}
    )
    d_model_candidates = [d for d in d_model_candidates if 64 <= d <= 512]

    search_space = {
        # === Capacity ===
        "d_model": tune.choice(d_model_candidates or [128, 256, 512]),
        "e_layers": tune.choice([2, 3]),      # paper: 2 encoder layers
        "d_layers": 1,                        # keep decoder shallow
        "n_heads": tune.choice([4, 8]),       # 8 heads matches paper

        # FFN width as independent choices (simpler than coupling to d_model)
        "d_ff": tune.choice([512, 1024, 2048, 4096]),

        # === Non-stationary projector ===
        "proj_hidden_dim": tune.choice([64, 128, 256]),
        "proj_hidden_layers": 2,             # fixed as in the paper

        # === Optimization & regularization ===
        "lr": tune.loguniform(3e-5, 3e-4),   # around 1e-4
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout": tune.uniform(0.05, 0.25),
        "grad_clip": tune.uniform(0.5, 2.0),

        # === Training control ===
        "batch_size": tune.choice([16, 32, 64]),
        "patience": tune.choice([5, 10, 15]),
        "moving_avg": tune.choice([1, 3, 5, 7]),

        # === Fixed to match your pipeline ===
        "seq_len": 1600,
        "horizon": 400,
        "loss": "MSE",
        "norm": True,
        "use_series_stationarization": True,
        "use_de_stationary_attention": True,
    }

    return search_space

def informer_searchspace(num_vars: int | None = None):
    """
    Informer search space for long-term multivariate forecasting.

    Based on:
    - 3-layer encoder + 2-layer decoder in the paper (we center around 2–4 enc layers)
    - d_model ≈ 512, heads ∈ {8,16}, dropout ≈ 0.1
    - Adam with lr ≈ 1e-4, batch size ≈ 32
    """

    # If you want to adapt d_model a bit to channel count you *can*,
    # but Informer uses a fairly large fixed model (512) across datasets.
    if num_vars is not None and num_vars > 0:
        # Very mild scaling, but keep it in [256, 512]
        base = min(512, max(256, 32 * ((num_vars + 31) // 32)))
        d_model_candidates = sorted({256, base, 512})
    else:
        d_model_candidates = [256, 512]

    search_space = {
        # === Capacity ===
        # Encoder depth: paper uses 3–4 layers; we keep it modest for your long sequences.
        "d_model": tune.choice(d_model_candidates),
        "e_layers": tune.choice([2, 3, 4]),       # encoder layers
        # Decoder layers usually 2 in the paper; keep it small but tunable if your impl uses d_layers
        "d_layers": tune.choice([1, 2]),

        # ProbSparse attention factor c (paper uses c=5)
        "factor": tune.choice([3, 5, 7]),

        # Feedforward width (paper uses inner size 2048 with d_model=512)
        "d_ff": tune.choice([512, 1024, 2048]),

        # Multi-head attention: 8 or 16 heads in the paper
        "n_heads": tune.choice([8, 16]),

        # Distilling: paper uses distilling; allow tuning it
        "distil": tune.choice([True, False]),

        # === Optimization & regularization ===
        # LR centered around 1e-4 with decay
        "lr": tune.loguniform(3e-5, 3e-4),
        "dropout": tune.uniform(0.05, 0.25),
        "weight_decay": tune.loguniform(1e-6, 1e-3),

        "batch_size": tune.choice([16, 32, 64]),

        # Activation is GELU in the official impl
        "activation": "gelu",

        # Learning-rate adjustment strategy – reuse your existing options
        "lradj": tune.choice(["type1", "type2", "cosine", "none"]),

        # === Fixed / pipeline-aligned ===
        "loss": "MSE",          # Informer uses MSE 
        "horizon": 400,         # adjust if your eval config uses something else
        "seq_len": 1600,        # aligned with your other models
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([5, 10, 15]),
        "moving_avg": tune.choice([1, 3, 5, 7]),

        # Gradient clipping for stability on long sequences
        "grad_clip": tune.uniform(0.5, 2.0),
    }

    return search_space

def crossformer_searchspace():
    """Enhanced Crossformer search space"""
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([128, 256, 512, 1024]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3, 4]),
        "n_heads": tune.choice([4, 8, 12, 16]),
        
        # Better regularization
        "dropout": tune.loguniform(1e-4, 0.3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Learning
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        
        # Model-specific
        "seg_len": tune.choice([8, 16, 32, 50, 100, 200]),  # More granular
        "factor": tune.randint(1, 12),
        
        # Fixed
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        
        # Additional
        "grad_clip": tune.uniform(0.5, 2.0),
    }
    return search_space


def itransformer_searchspace():
    """Enhanced iTransformer search space"""
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([128, 256, 512, 1024]),
        "d_model": tune.choice([64, 128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3, 4]),
        "n_heads": tune.choice([2, 4, 8, 12, 16]),
        
        # Regularization
        "dropout": tune.loguniform(1e-4, 0.3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Learning
        "lr": tune.loguniform(1e-5, 1e-2),
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        
        # Model-specific
        "period_len": tune.choice([8, 16, 32, 50, 100, 200]),
        
        # Fixed
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        
        # Additional
        "grad_clip": tune.uniform(0.5, 2.0),
    }
    return search_space


def dlinear_searchspace():
    """Enhanced DLinear search space"""
    search_space = {
        "batch_size": tune.choice([32, 64, 128, 256]),  # DLinear can handle larger batches
        "d_ff": tune.choice([64, 128, 256, 512, 1024]),
        
        # Learning parameters
        "lr": tune.loguniform(1e-5, 1e-1),  # DLinear can use higher LR
        "lradj": tune.choice(["type1", "type2", "cosine"]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        
        # Regularization
        "dropout": tune.loguniform(1e-4, 0.2),  # Less dropout needed for simpler model
        
        # Fixed
        "horizon": 400,
        "seq_len": 1600,
        "loss": "MAE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([5, 10, 15]),  # Can be more aggressive
        "moving_avg": tune.choice([1, 3, 5, 7, 13]),  # More options for linear model
        
        # Additional
        "grad_clip": tune.uniform(0.5, 1.5),
    }
    return search_space


def timemixer_searchspace():
    search_space = {
        # === Core architecture ===
        # d_model is small in paper (16 / 32 / 128 depending on dataset)
        "d_model": tune.choice([16, 32, 64, 128, 256]),
        
        # Number of PDM blocks L (paper uses L=2, ablations show benefits for L=2–4)
        "e_layers": tune.choice([1, 2, 3, 4]),

        # Number of scales M (paper uses M=1 for short-term, M=3 for long-term)
        "num_scales": tune.choice([1, 2, 3, 4]),

        # Feedforward width (not explicit; derive as multiples of d_model)
        "d_ff_mult": tune.choice([2, 4, 8, 16]),  # d_ff = d_model * multiplier

        # === Learning parameters ===
        # Paper uses ADAM LR=1e-2 or 1e-3 depending on dataset
        "lr": tune.loguniform(1e-4, 3e-2),

        # weight decay is not used, aber kann helfen
        "weight_decay": tune.loguniform(1e-8, 1e-3),

        # === Regularization ===
        "dropout": tune.uniform(0.0, 0.3),  # paper uses small dropout
        "moving_avg": tune.choice([1, 3, 5, 7]),  # based on appendix

        # === Optimization and stabilization ===
        "batch_size": tune.choice([16, 32, 64, 128]),
        "grad_clip": tune.uniform(0.5, 2.0),

        "down_sampling_window": tune.choice([2, 4, 8]),
        "channel_independence": tune.choice([True, False]),

        # === Fixed — based on your global pipeline ===
        "seq_len": 1600,
        "horizon": 400,
        "loss": "MSE",
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([10, 15, 20]),
    }

    return search_space

def etsformer_searchspace():
    """
    ETSformer search space for long-term multivariate forecasting.

    Paper defaults:
      - e_layers = 2, decoder_stacks = 2
      - d_model = 512, d_ff = 2048
      - n_heads = 8
      - K in {0,1,2,3}
      - L (lookback) in {96,192,336,720}
      - lr in [1e-5, 1e-3]
    We keep the architecture close to the paper and mainly tune capacity &
    optimization, while adapting to your global seq_len/horizon.
    """

    search_space = {
        # === Capacity / architecture ===
        # Keep depth paper-like; optionally allow 3 encoder layers
        "d_model": tune.choice([256, 512]),
        "d_ff": tune.choice([1024, 2048, 4096]),
        "e_layers": tune.choice([2, 3]),          # encoder layers
        "decoder_stacks": tune.choice([1, 2]),    # decoder stacks (paper: 2)
        "n_heads": 8,                             # fixed as in paper
        "conv_kernel_size": 3,                    # embedding kernel size

        # Number of frequencies for Frequency Attention
        "K": tune.choice([0, 1, 2, 3]),           # paper’s grid

        # === Optimization & regularization ===
        # Around paper grid {1e-3,...,1e-5}, but log-uniform
        "lr": tune.loguniform(3e-5, 3e-3),
        "dropout": tune.uniform(0.1, 0.3),        # paper ~0.2, small band
        "weight_decay": tune.loguniform(1e-7, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),  # paper 32

        # === Fixed / training control (aligned with your framework) ===
        "loss": "MSE",
        "horizon": 400,
        "seq_len": 1600,
        "norm": True,
        "num_epochs": config.max_epochs,
        "patience": tune.choice([5, 10, 15]),
        "moving_avg": tune.choice([1, 3, 5, 7]),
        "grad_clip": tune.uniform(0.5, 2.0),
    }

    return search_space



def assemble_setup(setup_name: str):
    """Assemble setup with improved search spaces"""
    # Base eval config
    base_eval_cfg = config.default_eval_config()

    # If your eval config exposes the number of variables/channels,
    # you can pass it into timesnet_searchspace for a slightly smarter prior.
    num_vars = None
    for key in ["num_features", "n_features", "n_vars", "num_vars"]:
        if key in base_eval_cfg:
            num_vars = base_eval_cfg[key]
            break

    setups = {
        "duet": (
            base_eval_cfg,
            config.model_config(
                "duet.DUET", decoder_input_required=False, has_loss_importance=True
            ),
            duet_searchspace(),
        ),
        "crossformer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.Crossformer.Model", "transformer_adapter"
            ),
            crossformer_searchspace(),
        ),
        "itransformer": (
            base_eval_cfg,
            config.model_config(
                "time_series_library.iTransformer", "transformer_adapter"
            ),
            itransformer_searchspace(),
        ),
        "timexer": (
            base_eval_cfg,
            config.model_config(
                "time_series_library.TimeXer",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            timexer_searchspace(),
        ),
        "dlinear": (
            base_eval_cfg,
            config.model_config(
                "time_series_library.DLinear", "transformer_adapter"
            ),
            dlinear_searchspace(),
        ),
        "timesnet": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.TimesNet.Model",
                "transformer_adapter",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            timesnet_searchspace(num_vars=num_vars),
        ),
        "timemixer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.TimeMixer.Model",
                "transformer_adapter",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            timemixer_searchspace(),
        ),
        "nonstationary_transformer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.Nonsstationary_Transformer.Model",
                "transformer_adapter",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            nonstationary_transformer_searchspace(num_vars=num_vars),
        ),
        "fedformer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.FEDformer.Model",
                "transformer_adapter",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            fedformer_searchspace(),
        ),
        "informer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.Informer.Model",
                "transformer_adapter",
                decoder_input_required=True,   # Informer uses encoder-decoder
                has_loss_importance=False,
            ),
            informer_searchspace(num_vars=num_vars),
        ),
        "autoformer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.Autoformer.Model",
                "transformer_adapter",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            autoformer_searchspace(num_vars=num_vars),
        ),
        "etsformer": (
            base_eval_cfg,
            config.model_config(
                "paper_icps.tslib.models.ETSformer.Model",
                "transformer_adapter",
                decoder_input_required=True,
                has_loss_importance=False,
            ),
            etsformer_searchspace(),
        ),
    }

    setup = setups.get(setup_name)
    if not setup:
        raise ValueError(f"Unknown setup name: {setup_name}")
    return setup