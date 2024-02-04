import argparse
import copy
import glob
import json
import shutil
from pathlib import Path
from typing import Tuple
from mlx.utils import tree_flatten

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import get_model_path
from utils import load, save_weights

def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize model to MLX format"
    )

    parser.add_argument("--model", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    return parser


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    quantized_config = copy.deepcopy(config)

    nn.QuantizedLinear.quantize_module(
        model, q_group_size, q_bits, 
        linear_class_predicate=lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != config['num_local_experts']
    )
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def convert(
    model: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
):
    print("[INFO] Loading")
    model_path = get_model_path(model)
    model, tokenizer = load(model_path)
    with open(model_path/"config.json", "r") as file:
        config = json.load(file)
        
    weights = dict(tree_flatten(model.parameters()))
    dtype = mx.float16 if quantize else getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, q_group_size, q_bits)

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    save_weights(mlx_path, weights)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    with open(mlx_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)



if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))
