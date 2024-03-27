import json
from pathlib import Path


def isfloat(num):
    try:
        float(num)
        return True
    except:
        return False


root_path = Path('/home/mindlab/humanObjectDetectionDataset')
dataset_path = root_path / "rawData"
instance_paths = dict([(str(i), p)
                       for i, p in enumerate(dataset_path.iterdir()) if p.is_dir() and not p.name == "_ignore"])

lines = [f'{key} {value.name}' for key, value in instance_paths.items()]
print("Found the following instances:")
print("==============================")
print('\n'.join(lines) + '\n')

instance_key = None
while instance_key not in instance_paths:
    instance_key = input("Instance key (index): ")

start_from_time = None
while not isfloat(start_from_time):
    start_from_time = input("New start from time for instance: ")
start_from_time = float(start_from_time)

meta_path = str((instance_paths[instance_key] / "meta.json").absolute())
meta_data = None

with open(meta_path) as f:
    meta_data = json.load(f)

meta_data["start_from_time"] = start_from_time
with open(meta_path, 'w') as f:
    json.dump(meta_data, f, indent=4)

print(
    f"Successfully updated start_from_time to {str(start_from_time)} for instance {str(instance_paths[instance_key].name)}")
