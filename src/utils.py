import os
import json


def save_params(root_path: str, **kwargs):
    with open(os.path.join(root_path, "model_params.json"), "w") as f:
        json.dump(kwargs, f)
    print(f"Model params saved in {root_path}")



