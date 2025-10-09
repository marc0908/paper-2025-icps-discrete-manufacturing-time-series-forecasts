max_epochs = 150


def default_eval_config():
    return {
        "metrics": "all",
        "strategy_args": {
            "horizon": 400,
            "tv_ratio": 0.8,
            "train_ratio_in_tv": 0.75,
            "seed": 2025,
        },
    }


def model_config(model_name, model_adaptor=None, **kwargs):
    defaults = {
        "decoder_input_required": True,
        "has_loss_importance": False,
        "input_sampling": 1,
    }

    model_config = {
        "models": [
            {
                "adapter": model_adaptor,
                "model_name": model_name,
                "model_hyper_params": None,
                **defaults,
                **kwargs,
            }
        ]
    }
    return model_config


def duet_previous_good_params():
    good_params = [
        {
            "CI": 1,
            "batch_size": 64,
            "d_ff": 256,
            "d_model": 256,
            "dropout": 0.00372870680567492,
            "e_layers": 3,
            "factor": 3,
            "fc_dropout": 0.005818882802174602,
            "horizon": 400,
            "k": 3,
            "loss": "MAE",
            "lr": 0.0003433121413584701,
            "lradj": "type3",
            "moving_avg": 1,
            "n_heads": 6,
            "norm": True,
            "num_epochs": 150,
            "num_experts": 9,
            "patch_len": 200,
            "patience": 10,
            "seq_len": 1600,
        },
        {
            "CI": 1,
            "batch_size": 64,
            "d_ff": 256,
            "d_model": 256,
            "dropout": 0.00372870680567492,
            "e_layers": 3,
            "factor": 3,
            "fc_dropout": 0.005818882802174602,
            "horizon": 400,
            "k": 3,
            "loss": "MAE",
            "lr": 0.0003433121413584701,
            "lradj": "type3",
            "moving_avg": 3,
            "n_heads": 6,
            "norm": True,
            "num_epochs": 150,
            "num_experts": 9,
            "patch_len": 200,
            "patience": 10,
            "seq_len": 1600,
        },
        {
            "CI": 1,
            "batch_size": 64,
            "d_ff": 256,
            "d_model": 256,
            "dropout": 0.00372870680567492,
            "e_layers": 3,
            "factor": 3,
            "fc_dropout": 0.005818882802174602,
            "horizon": 400,
            "k": 3,
            "loss": "MAE",
            "lr": 0.0003433121413584701,
            "lradj": "type1",
            "moving_avg": 1,
            "n_heads": 6,
            "norm": True,
            "num_epochs": 150,
            "num_experts": 9,
            "patch_len": 200,
            "patience": 10,
            "seq_len": 1600,
        },
        {
            "CI": 1,
            "batch_size": 64,
            "d_ff": 256,
            "d_model": 256,
            "dropout": 0.00372870680567492,
            "e_layers": 3,
            "factor": 3,
            "fc_dropout": 0.005818882802174602,
            "horizon": 400,
            "k": 3,
            "loss": "MAE",
            "lr": 0.0003433121413584701,
            "lradj": "type1",
            "moving_avg": 3,
            "n_heads": 6,
            "norm": True,
            "num_epochs": 150,
            "num_experts": 9,
            "patch_len": 200,
            "patience": 10,
            "seq_len": 1600,
        },
    ]
    return good_params


def crossformer_previous_good_params():
    good_params = [
        {
            "batch_size": 64,
            "d_ff": 128,
            "d_model": 256,
            "dropout": 0.12835056752683843,
            "e_layers": 3,
            "factor": 6,
            "horizon": 400,
            "loss": "MAE",
            "lr": 9.824131591363273e-05,
            "lradj": "type3",
            "moving_avg": 5,
            "n_heads": 15,
            "norm": True,
            "num_epochs": 150,
            "patience": 10,
            "seg_len": 200,
            "seq_len": 1600,
        }
    ]
    """
        {
          "batch_size": 32,
          "d_ff": 512,
          "d_model": 512,
          "dropout": 0.06234190438094256,
          "e_layers": 3,
          "factor": 10,
          "horizon": 400,
          "loss": "MAE",
          "lr": 4.234270751420961e-05,
          "lradj": "type3",
          "moving_avg": 1,
          "n_heads": 5,
          "norm": True,
          "num_epochs": 150,
          "patience": 10,
          "seg_len": 200,
          "seq_len": 1600
     }"""
    return good_params


def itransformer_previous_good_params():
    good_params = [
        {
            "batch_size": 32,
            "d_ff": 256,
            "d_model": 64,
            "dropout": 0.022579905865628927,
            "e_layers": 3,
            "horizon": 400,
            "loss": "MAE",
            "lr": 1.3627395149482503e-05,
            "lradj": "type1",
            "moving_avg": 1,
            "n_heads": 1,
            "norm": True,
            "num_epochs": 150,
            "patience": 10,
            "period_len": 100,
            "seq_len": 1600,
        }
    ]
    return good_params


def dlinear_previous_good_params():
    good_params = [
        {
            "batch_size": 32,
            "d_ff": 128,
            "d_model": 256,
            "dropout": 0.35052370372001956,
            "horizon": 400,
            "loss": "MAE",
            "lr": 0.0001377110270953509,
            "lradj": "type1",
            "moving_avg": 3,
            "norm": True,
            "num_epochs": 150,
            "patience": 10,
            # "period_len": 128,
            "seq_len": 1600,
        }
    ]
    return good_params


def patchtst_previous_good_params():
    good_params = [
        {
            "batch_size": 64,
            "d_ff": 128,
            "d_model": 64,
            "dropout": 0.036980722656229745,
            "e_layers": 3,
            "horizon": 400,
            "loss": "MAE",
            "lr": 0.002293829508811099,
            "lradj": "type1",
            "moving_avg": 1,
            "n_heads": 18,
            "norm": True,
            "num_epochs": 150,
            "patch_len": 16,
            "patience": 10,
            "seq_len": 1600,
        }
    ]
    return good_params


def pdf_previous_good_params():
    good_params = [
        {
            "batch_size": 64,
            "d_ff": 256,
            "d_model": 64,
            "dropout": 0.006535483804350568,
            "e_layers": 3,
            "fc_dropout": 0.47898612660442536,
            "horizon": 400,
            "kernel_list": [3, 5, 7, 11],
            "loss": "MAE",
            "lr": 0.00042214718658111313,
            "lradj": "type1",
            "moving_avg": 1,
            "n_head": 13,
            "period_len": [100],
            "patch_len": [20],
            "seq_len": 1600,
            "stride": [4],
            "train_epochs": 150,
        }
    ]
    return good_params


def previous_good_params(model_name):
    good_params = {
        "duet": duet_previous_good_params(),
        "crossformer": crossformer_previous_good_params(),
        "itransformer": itransformer_previous_good_params(),
        "dlinear": dlinear_previous_good_params(),
        "patchtst": patchtst_previous_good_params(),
        "pdf": pdf_previous_good_params(),
    }
    return good_params[model_name]
