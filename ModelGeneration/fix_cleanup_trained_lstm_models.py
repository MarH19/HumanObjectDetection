import json
import torch
from pathlib import Path

trained_models_path = Path(__file__).parent / "TrainedModels"

model_params_list = []
with open(str((trained_models_path / "RnnModelsParameters.json").absolute()), 'r') as f:
    model_params_list: list = json.load(f)

for p in trained_models_path.iterdir():
    if p.is_file() and p.name.startswith("LSTMModel_") and p.suffix == ".pth":
        if not any(mp["model_name"] == p.name.replace(".pth", "") for mp in model_params_list):
            print(
                f"no model hyperparameters found for model file {p.name} -> delete file")
            p.unlink()
        else:
            # fix parameter naming from "lstm.<...>" to "rnn_model.<...>"
            state_dict = torch.load(
                str(p.absolute()), map_location=torch.device('cpu'))
            for key in list(state_dict.keys()):
                new_key = key.replace("lstm.", "rnn_model.")
                state_dict[new_key] = state_dict.pop(key)
            torch.save(state_dict, str(p.absolute()))
