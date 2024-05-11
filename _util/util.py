import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def user_input_choose_from_list(choices: list[Any], caption: str, list_item_label_selector: Callable[[Any], str] | None = None):
    lines = [
        f'{i} {list_item_label_selector(v) if list_item_label_selector is not None else v}' for i, v in enumerate(choices)]
    print('\n' + '\n'.join(lines))
    index = None
    while index not in np.arange(0, len(choices), 1):
        index = int(input(f"Choose from {caption}: "))
    return choices[index]


def choose_robot_motion():
    repo_root_path = Path(__file__).parents[1]
    robot_motions_path = repo_root_path / "frankaRobot" / "robotMotionPoints"
    robot_motions = [p for _, p in enumerate(
        robot_motions_path.iterdir()) if p.is_file and p.suffix == ".csv"]
    return user_input_choose_from_list(robot_motions, "Robot Motions", lambda m: m.name)


def choose_dataset():
    processed_data_path = Path(os.environ.get(
        "DATASET_REPO_ROOT_PATH")) / "processedData"

    sub_repo = dict([(str(i), p) for i, p in enumerate(
        processed_data_path.iterdir()) if p.is_dir()])
    print("sub repo:")
    lines = [f'{key} {value.name}' for key, value in sub_repo.items()]
    print('\n'.join(lines) + '\n')
    subrepo_key = None
    while subrepo_key not in sub_repo:
        subrepo_key = input(
            "Which sub repo should be used? (choose by index): ")

    full_path = processed_data_path / sub_repo[subrepo_key]

    datasets = dict([(str(i), p) for i, p in enumerate(full_path.iterdir())
                     if p.is_file and p.name.startswith("x_") and p.suffix == ".npy" and "test" not in p.name])
    lines = [f'{key} {value.name}' for key, value in datasets.items()]

    print("Datasets:")
    print('\n'.join(lines) + '\n')
    dataset_key = None
    while dataset_key not in datasets:
        dataset_key = input(
            "Which dataset should be used? (choose by index): ")
    return sub_repo[subrepo_key], datasets[dataset_key]


def choose_normalization_mode():
    normalization_modes = [
        {"key": "", "caption": "No Normalization"},
        {"key": "S", "caption": "Standardization (mean/variance)"},
        {"key": "N", "caption": "Normalization (min/max)"}]
    return user_input_choose_from_list(normalization_modes, "Normalization Mode", lambda m: m["caption"])


def normalize_dataset(normalization_mode, data_train, data_test):
    scaler = None
    norm_mins, norm_maxes, norm_means, norm_vars = None, None, None, None

    def scale(scaler_type: type[StandardScaler | MinMaxScaler]):
        scaler = scaler_type()
        N_train, S, D = data_train.shape
        N_test = data_test.shape[0]
        data_train = scaler.fit_transform(
            data_train.reshape(N_train * S, D)).reshape(N_train, S, D)
        data_test = scaler.transform(data_test.reshape(
            N_test * S, D)).reshape(N_test, S, D)
        return scaler

    if normalization_mode == "S":
        scaler = scale(StandardScaler)
        norm_means, norm_vars = scaler.mean_.tolist(), scaler.var_.tolist()
    elif normalization_mode == "N":
        scaler = scale(MinMaxScaler)
        norm_mins, norm_maxes = scaler.data_min_.tolist(), scaler.data_max_.tolist()

    return data_train, data_test, norm_mins, norm_maxes, norm_means, norm_vars
