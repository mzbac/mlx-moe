import glob
import mlx.core as mx


def load_weights(model_path):
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights
