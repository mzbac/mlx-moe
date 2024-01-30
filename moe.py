import json
from pathlib import Path
from mlx_lm.utils import save_weights
from transformers import AutoTokenizer
from utils import load_weights
import mlx.nn as nn

base_model_path = Path("./models/mistralai/Mistral-7B-Instruct-v0.2")

with open(base_model_path / "config.json", "r") as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

experts_weights = [
    load_weights(base_model_path),
    load_weights(Path("./models/mistralai/Mistral-7B-v0.1")),
    load_weights(Path("./models/berkeley-nest/Starling-LM-7B-alpha")),
    load_weights(Path("./models/mistralai/Mistral-7B-Instruct-v0.1")),
]
# use first expert as base model
base_model_weights = experts_weights[0]

config["num_local_experts"] = 4
config["num_experts_per_tok"] = 2
config["model_type"] = "mixtral"

weights = {}
for n, v in base_model_weights.items():
    if 'mlp' not in n:
        weights[n] = v


expert_layers = []

for i in range(config['num_hidden_layers']):
    weights[f"model.layers.{i}.block_sparse_moe.gate.weight"] = nn.Linear(config['hidden_size'], config['num_local_experts'], bias=False).weight
    for idx, e_w in enumerate(experts_weights):
        weights[f"model.layers.{i}.block_sparse_moe.experts.{idx}.w1.weight"] = e_w[f"model.layers.{i}.mlp.gate_proj.weight"]
        weights[f"model.layers.{i}.block_sparse_moe.experts.{idx}.w2.weight"] = e_w[f"model.layers.{i}.mlp.down_proj.weight"]
        weights[f"model.layers.{i}.block_sparse_moe.experts.{idx}.w3.weight"] = e_w[f"model.layers.{i}.mlp.up_proj.weight"]

mlx_path = Path("mlx_moe")

tokenizer.save_pretrained(mlx_path)
save_weights(mlx_path,weights=weights)

with open(mlx_path / "config.json", "w") as fid:
    json.dump(config, fid, indent=4)
