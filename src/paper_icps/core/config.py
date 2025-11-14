from __future__ import annotations

from typing import Any, Dict, List, Optional


# Global training defaults
MAX_EPOCHS = 150
# Backwards-compatible alias
max_epochs = MAX_EPOCHS


def default_eval_config() -> Dict[str, Any]:
    """
    Default evaluation configuration shared across experiments.
    """
    return {
        "metrics": "all",
        "strategy_args": {
            "horizon": 400,
            "tv_ratio": 0.8,
            "train_ratio_in_tv": 0.75,
            "seed": 2025,
        },
    }


def model_config(
    model_name: str,
    model_adaptor: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Build a model configuration block in the format expected by the training
    and evaluation pipelines.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. "duet", "crossformer".
    model_adaptor : str, optional
        Name of the adaptor class/function wrapping the raw model.
    overrides : dict
        Arbitrary key/value pairs to override default flags or hyper-parameters.

    Returns
    -------
    dict
        Configuration dictionary with a single-model "models" list.
    """
    defaults = {
        "decoder_input_required": True,
        "has_loss_importance": False,
        "input_sampling": 1,
    }

    return {
        "models": [
            {
                "adapter": model_adaptor,
                "model_name": model_name,
                "model_hyper_params": None,
                **defaults,
                **overrides,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Previous "good" hyper-parameter presets
# ---------------------------------------------------------------------------

# Common defaults used by most models
BASE_DEFAULTS: Dict[str, Any] = {
    "horizon": 400,
    "loss": "MAE",
    "norm": True,
    "num_epochs": MAX_EPOCHS,
    "patience": 10,
    "seq_len": 1600,
}

# DUET shared base configuration
DUET_BASE = {
    **BASE_DEFAULTS,
    "CI": 1,
    "batch_size": 64,
    "d_ff": 256,
    "d_model": 256,
    "dropout": 0.00372870680567492,
    "e_layers": 3,
    "factor": 3,
    "fc_dropout": 0.005818882802174602,
    "k": 3,
    "lr": 0.0003433121413584701,
    "n_heads": 6,
    "num_experts": 9,
    "patch_len": 200,
}


PREVIOUS_GOOD_PARAMS: Dict[str, List[Dict[str, Any]]] = {
    "duet": [
        {**DUET_BASE, "lradj": "type3", "moving_avg": 1},
        {**DUET_BASE, "lradj": "type3", "moving_avg": 3},
        {**DUET_BASE, "lradj": "type1", "moving_avg": 1},
        {**DUET_BASE, "lradj": "type1", "moving_avg": 3},
    ],

    "crossformer": [
        {
            **BASE_DEFAULTS,
            "lradj": "type3",
            "batch_size": 64,
            "d_ff": 128,
            "d_model": 256,
            "dropout": 0.12835056752683843,
            "e_layers": 3,
            "factor": 6,
            "lr": 9.824131591363273e-05,
            "moving_avg": 5,
            "n_heads": 15,
            "seg_len": 200,
        }
        # If you want to re-enable alternative presets, add more dicts here.
    ],

    "itransformer": [
        {
            **BASE_DEFAULTS,
            "lradj": "type1",
            "batch_size": 32,
            "d_ff": 256,
            "d_model": 64,
            "dropout": 0.022579905865628927,
            "e_layers": 3,
            "lr": 1.3627395149482503e-05,
            "moving_avg": 1,
            "n_heads": 1,
            "period_len": 100,
        }
    ],

    "dlinear": [
        {
            **BASE_DEFAULTS,
            "lradj": "type1",
            "batch_size": 32,
            "d_ff": 128,
            "d_model": 256,
            "dropout": 0.35052370372001956,
            "lr": 0.0001377110270953509,
            "moving_avg": 3,
        }
    ],

    "patchtst": [
        {
            **BASE_DEFAULTS,
            "lradj": "type1",
            "batch_size": 64,
            "d_ff": 128,
            "d_model": 64,
            "dropout": 0.036980722656229745,
            "e_layers": 3,
            "lr": 0.002293829508811099,
            "moving_avg": 1,
            "n_heads": 18,
            "patch_len": 16,
        }
    ],

    "pdf": [
        {
            "batch_size": 64,
            "d_ff": 256,
            "d_model": 64,
            "dropout": 0.006535483804350568,
            "e_layers": 3,
            "fc_dropout": 0.47898612660442536,
            "kernel_list": [3, 5, 7, 11],
            "lr": 0.00042214718658111313,
            "lradj": "type1",
            "moving_avg": 1,
            "n_head": 13,
            "period_len": [100],
            "patch_len": [20],
            "stride": [4],
            "train_epochs": MAX_EPOCHS,
            # Keep pdf-specific keys as in the original config
            # Pdf also uses these generic forecasting settings
            "horizon": 400,
            "seq_len": 1600,
            "norm": True,
            "patience": 10,
        }
    ],
}


# ---------------------------------------------------------------------------
# Backwards-compatible helper functions
# ---------------------------------------------------------------------------

def duet_previous_good_params() -> List[Dict[str, Any]]:
    """Backwards-compatible wrapper for DUET presets."""
    return PREVIOUS_GOOD_PARAMS["duet"]


def crossformer_previous_good_params() -> List[Dict[str, Any]]:
    return PREVIOUS_GOOD_PARAMS["crossformer"]


def itransformer_previous_good_params() -> List[Dict[str, Any]]:
    return PREVIOUS_GOOD_PARAMS["itransformer"]


def dlinear_previous_good_params() -> List[Dict[str, Any]]:
    return PREVIOUS_GOOD_PARAMS["dlinear"]


def patchtst_previous_good_params() -> List[Dict[str, Any]]:
    return PREVIOUS_GOOD_PARAMS["patchtst"]


def pdf_previous_good_params() -> List[Dict[str, Any]]:
    return PREVIOUS_GOOD_PARAMS["pdf"]


def previous_good_params(model_name: str) -> List[Dict[str, Any]]:
    """
    Get the list of "previous good" hyper-parameter configurations
    for a given model name (e.g., "duet", "crossformer").
    """
    try:
        return PREVIOUS_GOOD_PARAMS[model_name]
    except KeyError as e:
        raise KeyError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {list(PREVIOUS_GOOD_PARAMS.keys())}"
        ) from e