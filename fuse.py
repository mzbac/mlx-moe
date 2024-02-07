import argparse
import glob
import json
import shutil
from pathlib import Path

from mlx.utils import tree_flatten, tree_unflatten
import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import apply_lora_layers, dequantize
from mlx_lm.utils import get_model_path,  upload_to_hub, save_weights as mlx_lm_save_weights
from utils import save_weights, load


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="lora_fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Path to the trained adapter weights (npz or safetensors).",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Path to the original Hugging Face model. Required for upload if --model is a local directory.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--de-quantize",
        help="Generate a de-quantized model.",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    print("Loading pretrained model")
    args = parse_arguments()

    model_path = get_model_path(args.model)
    model, tokenizer = load(model_path)
    with open(model_path/"config.json", "r") as file:
        config = json.load(file)

    model.freeze()
    model = apply_lora_layers(model, args.adapter_file)
    
    fused_linears = [
        (n, m.to_linear())
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]

    model.update_modules(tree_unflatten(fused_linears))

    if args.de_quantize:
        print("De-quantizing model")
        model = dequantize(model)

    weights = dict(tree_flatten(model.parameters()))

    save_path = Path(args.save_path)

    if not args.de_quantize:
        mlx_lm_save_weights(save_path, weights)
    else:
        save_weights(save_path, weights)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, save_path)

    tokenizer.save_pretrained(save_path)

    if args.de_quantize:
        config.pop("quantization", None)

    with open(save_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    if args.upload_repo is not None:
        hf_path = args.hf_path or (
            args.model if not Path(args.model).exists() else None
        )
        if hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        upload_to_hub(args.save_path, args.upload_repo, hf_path)


if __name__ == "__main__":
    main()