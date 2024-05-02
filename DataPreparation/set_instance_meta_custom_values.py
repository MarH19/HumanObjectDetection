import json
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def isfloat(num):
    try:
        float(num)
        return True
    except:
        return False


# DATASET_REPO_ROOT_PATH=<absolute-path-to-dataset-repo-root-folder>
dataset_repo_root_path = Path(os.environ.get("DATASET_REPO_ROOT_PATH"))
dataset_path = dataset_repo_root_path / "rawData"
instance_paths = dict([(str(i), p)
                       for i, p in enumerate(dataset_path.iterdir()) if p.is_dir() and not p.name == "_ignore"])

lines = [f'{key} {value.name}' for key, value in instance_paths.items()]
print("Found the following instances:")
print("==============================")
print('\n'.join(lines) + '\n')

instance_key = None
while instance_key not in instance_paths:
    instance_key = input("Instance key (index): ")

meta_path = str((instance_paths[instance_key] / "meta.json").absolute())
meta_data = None

with open(meta_path) as f:
    meta_data = json.load(f)

current_start_from_time = meta_data["start_from_time"]
current_reference_duration_multiplier_lower = meta_data[
    "reference_duration_multiplier_lower"] if "reference_duration_multiplier_lower" in meta_data else None
current_reference_duration_multiplier_upper = meta_data[
    "reference_duration_multiplier_upper"] if "reference_duration_multiplier_upper" in meta_data else None

start_from_time = None
while not isfloat(start_from_time):
    start_from_time = input(
        f"New start from time for instance (current: {current_start_from_time}) (default is -1): ")
start_from_time = float(start_from_time)

reference_duration_multiplier_lower = None
has_input = False
while not isfloat(reference_duration_multiplier_lower) and reference_duration_multiplier_lower != "":
    reference_duration_multiplier_lower = input(
        f"New custom multiplier for min/max time-window length lower bound (current: {current_reference_duration_multiplier_lower}) (empty + enter for default): ")
reference_duration_multiplier_lower = float(reference_duration_multiplier_lower) if isfloat(
    reference_duration_multiplier_lower) else None

reference_duration_multiplier_upper = None
has_input = False
while not isfloat(reference_duration_multiplier_upper) and reference_duration_multiplier_upper != "":
    reference_duration_multiplier_upper = input(
        f"New custom multiplier for min/max time-window length upper bound (current: {current_reference_duration_multiplier_upper}) (empty + enter for default): ")
reference_duration_multiplier_upper = float(
    reference_duration_multiplier_upper) if isfloat(reference_duration_multiplier_upper) else None

meta_data["start_from_time"] = start_from_time
meta_data["reference_duration_multiplier_lower"] = reference_duration_multiplier_lower
meta_data["reference_duration_multiplier_upper"] = reference_duration_multiplier_upper
with open(meta_path, 'w') as f:
    json.dump(meta_data, f, indent=4)

print(
    f"Successfully updated custom meta values for instance {str(instance_paths[instance_key].name)}")
