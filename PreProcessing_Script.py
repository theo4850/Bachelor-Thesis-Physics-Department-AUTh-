import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

Model = {
  "cnn_":         {"preprocess": "baseline"},
  "cnn_aug":     {"preprocess": "baseline"},
  "vgg16_frozen":       {"preprocess": "vgg16"},
  "vgg16_frozen_aug":   {"preprocess": "vgg16"},
  "vgg16_finetune":     {"preprocess": "vgg16"},
  "vgg16_finetune_aug": {"preprocess": "vgg16"},
}

def grayscale_to_rgb_float32(images:np.ndarray) -> np.ndarray:
  if images.ndim == 3:
    images = np.expand_dims(images, -1)
  if images.shape[-1] == 1:
    images = np.repeat(images, 3, axis=-1)
  return images.astype(np.float32)

def preprocessing(images: np.ndarray, preprocess_name: str) -> np.ndarray:
  images = grayscale_to_rgb_float32(images)
  if preprocess_name == "baseline":
    return images/255.0
  elif preprocess_name == "vgg16":
    return vgg16_preprocess(images)
  raise ValueError(f"Unknown preprocess: {preprocess_name}")

def _infer_key(model_or_name):
  if isinstance(model_or_name, str):
    return model_or_name

  if callable(model_or_name) and hasattr(model_or_name, "__name__"):
    name = model_or_name.__name__
    return name.replace("model_", "")

  name = getattr(model_or_name, "name", None)
  if isinstance(name, str) and len(name) > 0:
    return name

  raise ValueError("Δώσε string ή model-builder function ή keras Model instance.")


def dataset_preparation(model_or_name: str, x_train_val: np.ndarray, x_test: np.ndarray):
  key = _infer_key(model_or_name)
  
  if key not in Model:
    raise ValueError(f"Unknown model: {key}")
  pp = Model[key]["preprocess"]
  print("Using preprocessing:", pp)
  x_train_val_pp = preprocessing(x_train_val, pp)
  x_test_pp = preprocessing(x_test, pp)
  return x_train_val_pp, x_test_pp