import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import get_model_path, make_shards
from mlx_lm.tuner.utils import apply_lora_layers, tree_unflatten
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from phi2moe import ModelArgs, Model


def load_weights(model_path: str):
    model_path = get_model_path(model_path)
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load(
    path_or_hf_repo: str, tokenizer_config={}, adapter_file: str = None
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path)
    if adapter_file is not None:
        model = apply_lora_layers(model, adapter_file)
        adapters = list(mx.load(adapter_file).items())
        model.update(tree_unflatten(adapters))
        model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
    return model, tokenizer


def load_model(model_path: Path) -> nn.Module:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            quantization = config.get("quantization", None)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model = Model(ModelArgs.from_dict(config))

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
            and m.weight.shape[0]
            != config[
                "num_local_experts"
            ],  # avoid quantizing gate layers, otherwise we have to re-quant and upload all the mixtral models
        )

    # print(weights.keys())
    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())

    model.eval()
    return model


def fetch_from_hub(
    model_path: Path,
) -> Tuple[Dict, dict, PreTrainedTokenizer]:
    model = load_model(model_path)

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, config.to_dict(), tokenizer

def save_weights(save_path: Union[str, Path], weights: Dict[str, Any]) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = "model-{:05d}-of-{:05d}.safetensors" if shards_count > 1 else "model.safetensors"

    index_data = {"metadata": {"total_size": 0}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name
        
        mx.save_safetensors(str(shard_path), shard)

        for tensor_name in shard.keys():
            index_data["weight_map"][tensor_name] = shard_name

        index_data["metadata"]["total_size"] += shard_path.stat().st_size

    sorted_weight_map = {k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])}
    
    with open(save_path / 'model.safetensors.index.json', 'w') as f:
        json.dump({"metadata": index_data["metadata"], "weight_map": sorted_weight_map}, f, indent=4)
