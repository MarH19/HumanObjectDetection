import re
import torch
from pathlib import Path

trained_models_path = Path(__file__).parent / "TrainedModels"

original_pattern = r'rnn_model\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)'
replacement_pattern = replacement = r'rnn_layers.\2.\1_l0'

for p in trained_models_path.iterdir():
    if p.is_file() and p.suffix == ".pth" and (p.name.startswith("LSTMModel") or p.name.startswith("GRUModel")):
        state_dict = torch.load(
            str(p.absolute()), map_location=torch.device('cpu'))
        for key in list(state_dict.keys()):
            new_key = re.sub(original_pattern, replacement_pattern, key)
            state_dict[new_key] = state_dict.pop(key)
        torch.save(state_dict, str(p.absolute()))
