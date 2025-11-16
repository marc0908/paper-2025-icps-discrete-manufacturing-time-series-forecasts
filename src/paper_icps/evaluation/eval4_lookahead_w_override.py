import numpy as np

from ..core import common, config
import eval_common

def find_trajectory_override_ranges(arr):
    override_val = np.max(arr)
    found_len = 0
    found_len_start = 0
    overrides = []

    for i in range(len(arr)):
        if abs(arr[i] - override_val) < 1e-5:
            if found_len == 0:
                found_len_start = i
            found_len += 1
        elif found_len > 0:
            overrides.append((found_len_start, i))
            found_len = 0
    return overrides


def eval_model(modelname, modelpath, data, n_runs):
    model = common.restore_model(modelpath)

    print("========= ", model_name)
    print("Model scaler (must match):", model.scaler)
    print("Model parameter count: ", common.sum_model_params(model))

    lookback_len = model.config.seq_len
    generated_len_per_run = model.config.horizon

    n_recursion = 1

    cfg = config.default_eval_config()
    seed = cfg["strategy_args"]["seed"]
    common.set_fixed_seed(seed)
    rng = np.random.default_rng(seed=seed)
    print("Seed:", seed)

    train_data, valid_data, test_data = common.split_data(
        data,
        cfg["strategy_args"]["tv_ratio"],
        cfg["strategy_args"]["train_ratio_in_tv"],
    )

    override_ranges = find_trajectory_override_ranges(test_data["Override"])
    print("Available overrides in test data:", len(override_ranges), ", using:", n_runs)

    stats_avgs = []
    stats_per_variable = []
    trajectory_starts = []
    generated_results = []

    for override_start, override_end in override_ranges[:n_runs]:
        offset = 4  # timesteps, required as otherwise the new target pitch is not in the input data
        trajectory_start = override_start - (lookback_len - offset)
        if trajectory_start < 0:
            print("Skipping ", override_start, " ... lookback incomplete")
            continue

        trajectory_starts.append(trajectory_start)
        in_data = test_data[trajectory_start:].values

        if test_data["Override"][trajectory_start + lookback_len - 1] != 1:
            print("Warning: Override not in lookback, no forecast possible.")
            breakpoint()

        tdelta = 5
        override_start_range = test_data["TargetPitch"][
            trajectory_start
            + lookback_len
            - tdelta : trajectory_start
            + lookback_len
            + tdelta
        ]
        diffs = np.abs(np.diff(override_start_range))
        expected_override_start_idx = 2
        if np.sum(diffs[expected_override_start_idx + 1 :]) != 0:
            print(
                "Warning: No indication for override in TargetPitch data (Pitch should remain steady)"
            )
            print(
                test_data["Override"][
                    trajectory_start
                    + lookback_len
                    - tdelta : trajectory_start
                    + lookback_len
                    + tdelta
                ]
            )
            print(override_start_range)

        stat, generated = eval_common.forecast_and_stats(model, in_data, n_recursion)
        generated_results.append(generated)
        stats_avgs.append(stat["avg"])
        stats_per_variable.append(stat["per_variable"])

    stats_avgs = np.vstack(stats_avgs)
    stats_per_variable = np.vstack(stats_per_variable)

    lowest_mae, lowest_mse = stats_avgs.min(axis=0)
    lowest_idx = stats_avgs.argmin(axis=0)[0]

    highest_mae, highest_mse = stats_avgs.max(axis=0)
    highest_idx = stats_avgs.argmax(axis=0)[0]

    print("Lowest MAE:", lowest_mae, "MSE:", lowest_mse, "at index:", lowest_idx)
    print("Highest MAE:", highest_mae, "MSE:", highest_mse, "at index:", highest_idx)

    stats_avgs_sorted_idx = stats_avgs[:, 0].argsort()
    stats_avgs_len = stats_avgs.shape[0]

    lower_quartile_idx = stats_avgs_sorted_idx[int(stats_avgs_len * 0.25)]
    lower_quartile_mae = stats_avgs[lower_quartile_idx, :][0]
    upper_quartile_idx = stats_avgs_sorted_idx[int(stats_avgs_len * 0.75)]
    upper_quartile_mae = stats_avgs[upper_quartile_idx, :][0]

    eval_common.plot_generated_tight(
        test_data,
        generated_results[lower_quartile_idx],
        trajectory_starts[lower_quartile_idx],
        lookback_len,
        generated_len_per_run=generated_len_per_run,
        fname=f"eval4_{modelname}_lower_mae_idx.pdf",
        gen_color="lime",
        show_xlabel=False,
        desc=f"MAE={lower_quartile_mae:.3f}",
        markOverride=True,
    )
    eval_common.plot_generated_tight(
        test_data,
        generated_results[upper_quartile_idx],
        trajectory_starts[upper_quartile_idx],
        lookback_len,
        generated_len_per_run=generated_len_per_run,
        fname=f"eval4_{modelname}_higher_mae_idx.pdf",
        showtitles=False,
        desc=f"MAE={upper_quartile_mae:.3f}",
        markOverride=True,
    )

    return eval_common.create_mae_mse_table_row(
        test_data.columns, stats_per_variable, stats_avgs
    )


if __name__ == "__main__":
    args = eval_common.parse_args()

    data_w_overrides = common.load_csv(args.data_path)
    results = {}

    for model_name, model_path in args.model:
        result_row = eval_model(model_name, model_path, data_w_overrides, args.nruns)
        results[model_name] = result_row

    eval_common.print_latex_table_average_only(data_w_overrides, results)
