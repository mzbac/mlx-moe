from huggingface_hub import HfApi, ModelCard, logging

logging.set_verbosity_info()

api = HfApi()
api.create_repo(repo_id="mzbac/qwen-1.5-2x3-hf", exist_ok=True)
api.upload_folder(
    folder_path='lora_fused_model',
    repo_id="mzbac/qwen-1.5-2x3-hf",
    repo_type="model",
)