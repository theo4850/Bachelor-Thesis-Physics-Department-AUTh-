import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

Model = {
  "cnn":         {"preprocess": "baseline"},
  "cnn_aug":     {"preprocess": "baseline"},
  "effnetb0_frozen":       {"preprocess": "effnet"},
  "effnetb0_frozen_aug":   {"preprocess": "effnet"},
  "effnetb0_finetune":     {"preprocess": "effnet"},
  "effnetb0_finetune_aug": {"preprocess": "effnet"},
}

def grayscale_to_rgb_float32(images:np.ndarray) -> np.ndarray:
  if images.ndim == 3:
    images = np.expand_dims(images, -1)
  if images.shape[-1] == 1:
    images = np.repeat(images, 3, axis=-1)
  return images.astype(np.float32)

def preprocessing(images: np.ndarray, preprocess_name: str) -> np.ndarray:
  if preprocess_name == "baseline":
    return images/255.0
  elif preprocess_name == "effnet":
    images = grayscale_to_rgb_float32(images)
    return effnet_preprocess(images)    
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