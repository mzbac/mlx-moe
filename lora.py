import random
from typing import Tuple
from mlx_lm import load
from mlx_lm.lora import LoRALinear
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import TrainingArgs, train
import mlx.optimizers as optim

import json
from pathlib import Path


class Dataset:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

def load_dataset(path: str, train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as fid:
        file_content = fid.read()
        data = json.loads(file_content)

    # Combine instruction and output into the desired format
    combined_data = [f'[INST] {{ {item["instruction"]} }} [/INST] [INST] {{ {item["output"]} }} [/INST]' for item in data]

    random.shuffle(combined_data)

    split_idx = int(len(combined_data) * train_split)
    train_data = combined_data[:split_idx]
    val_data = combined_data[split_idx:]

    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)

    return train_dataset, val_dataset


def main():
    train_dataset_path = "./data/WizardLM/WizardLM_evol_instruct_70k/alpaca_evol_instruct_70k.json"

    model_path = "../mlx-moe/mlx_model/"

    model, tokenizer = load(model_path)

    train_dst, valid_dst = load_dataset(train_dataset_path)
    # train_dst, valid_dst = train_dst[:1], valid_dst[:1]     
    model.freeze()
    for l in model.model.layers:
        l.self_attn.q_proj = LoRALinear.from_linear(
            l.self_attn.q_proj, r=16, lora_alpha=32, lora_dropout=0.1
        )
        l.self_attn.v_proj = LoRALinear.from_linear(
            l.self_attn.v_proj, r=16, lora_alpha=32, lora_dropout=0.1
        )
        # l.self_attn.o_proj = LoRALinear.from_linear(l.self_attn.o_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(
                l.block_sparse_moe.gate, r=16, lora_alpha=32, lora_dropout=0.1
            )

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")
   
    trainingArgs = TrainingArgs(
        batch_size=2,
        iters=9000,
        val_batches=1,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=200,
        adapter_file="adapters.npz",
        max_seq_length=2048,
    )

    model.train()
    opt = optim.AdamW(learning_rate=1e-5)

    train(
        model=model,
        tokenizer=tokenizer,
        args=trainingArgs,
        optimizer=opt,
        train_dataset=train_dst,
        val_dataset=valid_dst,
    )


main()