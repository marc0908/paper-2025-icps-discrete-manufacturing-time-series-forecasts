import argparse
import os
import shutil
import csv
import datetime
import glob
import pprint


def find_min_val_loss(progress_csv):
    min_val_loss = float("inf")

    with open(progress_csv, mode="r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            val_loss = float(row[0])

            if val_loss < min_val_loss:
                min_val_loss = val_loss

    return min_val_loss


def find_best_model(parent_folder):

    best_min_val_loss = float("inf")
    best_model_dir = None

    for dir_name in os.listdir(parent_folder):
        dir_path = os.path.join(parent_folder, dir_name)

        if os.path.isdir(dir_path):
            progress_csv = os.path.join(dir_path, "progress.csv")

            if os.path.exists(progress_csv):
                min_val_loss = find_min_val_loss(progress_csv)

                if min_val_loss < best_min_val_loss:
                    best_model_dir = dir_name
                    best_min_val_loss = min_val_loss
    return best_min_val_loss, best_model_dir


def archive_model(parent_dir, model_dir, output_folder):
    src_dir_path = os.path.join(parent_dir, model_dir)
    short_model_dir, _ = model_dir.split(
        "batch_size=", 1
    )  # not ideal, but works for now ...
    short_model_dir = short_model_dir.strip("_")
    dest_dir_path = os.path.join(output_folder, short_model_dir)
    if os.path.exists(dest_dir_path):
        print(f"-> existing {dest_dir_path}")
    else:
        shutil.copytree(src_dir_path, dest_dir_path)
        print(f"-> archived to {dest_dir_path}")
    return dest_dir_path


def parse_env_file(env_path):
    env_vars = {}
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not "=" in line:
                continue
            key, path = line.split("=", 1)
            env_vars[key] = path
    return env_vars


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process model result folders and extract checkpoints from env."
    )

    parser.add_argument(
        "--model",
        nargs=2,
        action="append",
        metavar=("MODEL_ENV_VAR", "MODEL_PATH"),
        required=True,
        help="Specify model as a pair: ENV_VAR_NAME FOLDER_PATH (can be repeated)",
    )

    parser.add_argument(
        "--update-env-file",
        type=str,
        required=False,
        help="Path to shell-style env file with checkpoint paths",
    )

    parser.add_argument(
        "--archive-path",
        type=str,
        default=None,
        required=True,
        help="Path to archive dir",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Run in dry mode (do not perform any actual processing)",
    )

    return parser.parse_args()


def write_updated_env_file(env_path, merged_checkpoints):
    existing = []
    with open(env_path, "r") as f:
        for l in f.readlines():
            existing.append(l)
    print(existing)
    with open(env_path, "w") as f:
        f.write("#!/bin/bash\n")

        for var_name, checkpoint_path in merged_checkpoints.items():
            f.write(f"{var_name}={checkpoint_path}\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        f.write(f"# Previous entries (before {timestamp}):\n")
        for l in existing:
            if l[0] != "#":
                f.write("#")
            f.write(l.strip())
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    parent_folder = "~/ray_results/"
    parent_subfolders = [
        "patchtst_w_overrides_2025-04-10_14-32-54/",
        "pdf_w_overrides_2025-04-09_22-10-46/",
        "dlinear_w_overrides_individual_flag_2025-04-07_20-06-55",
        "itransformer_w_overrides_2025-04-06_18-29-52",
        "duet_w_overrides2_2025-03-28_03-28-06",
        "crossformer_w_overrides_2025-03-29_04-20-33",
    ]
    output_folder = "./results/"

    args = parse_args()

    archived_model_checkpoints = {}

    for env_path_var_name, results_path in args.model:
        print(f"Checking {results_path}")

        best_min_val_loss, best_model_dir = find_best_model(results_path)
        if best_model_dir:
            print(f"-> Found best in {results_path} with loss {best_min_val_loss}")
            if args.dry:
                continue
            archive_dir = archive_model(results_path, best_model_dir, args.archive_path)
            for archived_checkpoint in glob.glob(
                os.path.join(archive_dir, "checkpoint*/checkpoint.pt")
            ):
                archived_model_checkpoints[env_path_var_name] = (
                    f'"{archived_checkpoint}"'
                )
        else:
            print(f"-> None found in {results_path}.")

    pprint.pprint(archived_model_checkpoints)

    if args.update_env_file:
        env_file_checkpoints = parse_env_file(args.update_env_file)
        pprint.pprint(env_file_checkpoints)
        env_file_checkpoints.update(archived_model_checkpoints)
        print("Merged checkpoints:")
        pprint.pprint(env_file_checkpoints)
        if not args.dry:
            write_updated_env_file(args.update_env_file, env_file_checkpoints)
