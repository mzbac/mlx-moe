import json
from pathlib import Path
from mlx_lm.utils import  get_model_path
from transformers import AutoTokenizer
from utils import load_weights, save_weights
import mlx.nn as nn

# Paths for expert models, the first one is also used as the base model
EXPERT_MODEL_PATHS = [
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-7B",
    "Crystalcareai/CrystalQwen-1.5-7B"
]

# Update configuration based on the number of expert models
config_update = {
    "num_local_experts": len(EXPERT_MODEL_PATHS),
    "num_experts_per_tok": 2,
    "model_type": "qwen2-moe",
    "architectures": ["Qwen2MoeForCausalLM"],
}

MLX_SAVE_PATH = Path("mlx_moe")


def load_config(path):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except IOError as e:
        print(f"Error reading file {path}: {e}")
        raise


def save_config(config, path):
    try:
        with open(path, "w") as file:
            json.dump(config, file, indent=4)
    except IOError as e:
        print(f"Error writing file {path}: {e}")
        raise


def update_weights(expert_weights, config):
    weights = {}
    for n, v in expert_weights[0].items():  # Use the base model (first expert) weights
        if "mlp" not in n:
            weights[n] = v

    for i in range(config["num_hidden_layers"]):
        # Initialize gate weights
        weights[f"model.layers.{i}.block_sparse_moe.gate.weight"] = nn.Linear(
            config["hidden_size"], config["num_local_experts"], bias=False
        ).weight

        for idx, e_w in enumerate(expert_weights):
            base_path = f"model.layers.{i}.block_sparse_moe.experts.{idx}"
            weights[f"{base_path}.gate_proj.weight"] = e_w[
                f"model.layers.{i}.mlp.gate_proj.weight"
            ]
            weights[f"{base_path}.down_proj.weight"] = e_w[
                f"model.layers.{i}.mlp.down_proj.weight"
            ]
            weights[f"{base_path}.up_proj.weight"] = e_w[
                f"model.layers.{i}.mlp.up_proj.weight"
            ]


    return weights


def main():
    expert_weights = [load_weights(path) for path in EXPERT_MODEL_PATHS]

    config = load_config(get_model_path(EXPERT_MODEL_PATHS[0]) / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(get_model_path(EXPERT_MODEL_PATHS[0]))

    config.update(config_update)

    weights = update_weights(expert_weights, config)

    tokenizer.save_pretrained(MLX_SAVE_PATH)
    save_weights(MLX_SAVE_PATH, weights=weights)
    save_config(config, MLX_SAVE_PATH / "config.json")


if __name__ == "__main__":
    main()
