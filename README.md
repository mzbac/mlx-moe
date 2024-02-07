# mlx-moe

This repository contains scripts to create MoE models from pretrained llama/mistral models.

## Installation

Install the required dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Using the Script

**Configure Model Paths**: Open the `moe.py` script and modify the `EXPERT_MODEL_PATHS` list at the top of the script with the paths to your model weights and configurations. The first path in this list should be your base model.

**Execute the Script**: Run the script to create the MoE model from the command line.
```python
python moe.py
```
**Output**: The script will generate a new directory, `mlx_moe`, containing the tokenizer, updated weights, and configuration for the MoE model.

## MoE Gate Fine-tuning

After merging the MoE model, fine-tuning is crucial for optimal performance. The gate weights are initialized uniformly, and fine-tuning them is necessary for the model to learn how to effectively utilize the different experts.

**Quantization**: Given the potentially large size of the MoE model, you may want to quantize the model to reduce its size. Before starting to fine-tune the model, you can quantize it using the `quantize.py` script. This script will quantize the model and save it to the `mlx_model` directory.
```python
python quantize.py --model mlx_moe -q
```
**Download the Training Dataset**: Use the `download_dataset.py` script to download the training dataset. This script will download the dataset and save it to the `data` directory.
```python
python download_dataset.py WizardLM/WizardLM_evol_instruct_70k
```
**Fine-tuning Script**: Use the `lora.py` script for this purpose. This script applies the LoRA to fine-tune the gate and Q,V projection weights of the MoE model. Adjust the script accordingly based on your needs.
```python
python lora.py
```

**Inference**: Use the `inference.py` script to run inference on the fine-tuned model. This script will load the model and tokenizer from the `mlx_model` directory and run inference on the given input prompt.
```python
python inference.py --model mlx_model --adapter-file adapters.npz --prompt "Instruct: how backpropagation works.\nOutput:" -m 2000
```

## Fuse model
you can fuse the model with lora adapters using `fuse.py` script, and you can use llama.cpp/gptq/aws to quantize the fused model and use it in your application.
```python
python fuse.py --model mlx_model --adapter-file adapters.npz --de-quantize
```
To see supported options run:
```
python fuse.py --help
```