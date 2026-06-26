# Note

Root `AGENTS.md` is now the main Codex instruction entrypoint for this repository.
Project-specific memory and workflow files now live under `.agents/memory/`.
This legacy file is preserved as supplemental implementation guidance; do not delete its content.

# Coding Guidelines and Agent Rules

This document establishes development standards, code quality checks, and performance optimization guidelines for PyTorch, OpenCV, and GPU hardware execution within this repository.

---

## 1. PyTorch Best Practices

### 1.1 Device-Agnostic Execution with CPU Fallback
Always write code that runs seamlessly on CPU, CUDA (Nvidia GPU), or MPS (Apple Silicon if applicable). Use the global standard device mapping:
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
When loading checkpoint files or models, always use `map_location=device` or `map_location=torch.device('cpu')` to prevent memory allocation crashes:
```python
state_dict = torch.load(model_path, map_location=device)
```

To support resource-constrained farm deployments, all heavy GPU-dependent pipelines must implement robust exception handling and fallback gracefully to CPU execution:
```python
try:
    # Attempt to load model to CUDA device
    model.to(device)
except RuntimeError as e:
    if "CUDA" in str(e):
        device = torch.device("cpu")
        model.to(device)
```

### 1.2 Evaluation & Inference Mode
* Avoid using `model.eval()` without context managers.
* For inference, prefer using `torch.inference_mode()` (available in PyTorch 1.9+) as it is faster and uses less memory than `torch.no_grad()`.
```python
@torch.inference_mode()
def run_model_inference(model, inputs):
    return model(inputs.to(device))
```

### 1.3 Float Precision and Automatic Mixed Precision (AMP)
During model evaluation and training, utilize AMP (`torch.cuda.amp.autocast`) to speed up processing and reduce memory requirements on supported GPUs:
```python
with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and device.type == 'cuda'):
    outputs = model(inputs)
```

---

## 2. OpenCV Best Practices

### 2.1 Color Space Conversions
* OpenCV reads and writes images in **BGR** format by default.
* PyTorch, torchvision, and Pillow expect **RGB** format.
* Always convert color spaces explicitly when shifting between OpenCV and other libraries:
```python
import cv2

image_bgr = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```

### 2.2 Resource Cleanup & Context Management
* Always release resources (e.g. video files, camera streams, windows) explicitly when exceptions occur.
* Use `try...finally` blocks:
```python
cap = cv2.VideoCapture(video_path)
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # process frame
finally:
    cap.release()
    cv2.destroyAllWindows()
```

---

## 3. GPU & Memory Optimization

### 3.1 DataLoader Settings
* Pin memory to improve data transfer speed from CPU to GPU:
  `pin_memory=True` in `torch.utils.data.DataLoader` instantiation.
* Adjust `num_workers` depending on CPU cores, avoiding setting it too high on Windows environments to prevent sub-process overhead.

### 3.2 Freeing GPU Memory
To prevent Out-Of-Memory (OOM) errors during inference sequences:
* Delete unused tensors and call garbage collection when necessary.
* Empty the PyTorch CUDA cache between runs if running multiple heavy operations:
```python
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## 4. NumPy and Matrix Shape Checks (2D to 3D Projections)

As the project performs numerous coordinate projections and 3D matrix math transformations (e.g. using `inverse_intrinsic.npy`, `rot.npy`, and `depth_scale.npy` files), strict validation of type and shape is mandatory to prevent runtime shape mismatch errors:

### 4.1 Type Annotation & Shape Documentation
* Annotate NumPy array inputs with type hints (using `numpy.typing.NDArray`).
* Document the expected dimensions in docstrings or comments (e.g. `[N, 3]`, `[3, 3]`).

### 4.2 Matrix Dimension Verification
* Use explicit shape assertions before passing arrays to matrix multiplications or projections:
```python
import numpy as np
from numpy.typing import NDArray

def project_2d_to_3d(
    points_2d: NDArray[np.float32], 
    inv_intrinsic: NDArray[np.float32]
) -> NDArray[np.float32]:
    # Assert expected dimensions: points_2d shape [N, 2] or [N, 3], inv_intrinsic shape [3, 3]
    assert points_2d.ndim == 2, f"Expected 2D points array, got shape {points_2d.shape}"
    assert inv_intrinsic.shape == (3, 3), f"Expected 3x3 matrix, got shape {inv_intrinsic.shape}"
    
    # Process projection
    ...
```

---

## 5. Code Standards & Quality Checks
* **Formatters & Linters:** The codebase strictly adheres to formatting rules defined by `ruff`. Ensure any modifications are compliant:
  ```bash
  ruff check src main.py tools tests
  ```
* **Type-Checking:** Use `mypy` to statically verify all type definitions and function signatures across the repository:
  ```bash
  mypy src
  ```
* **Testing:** Write unit tests for new utility functions or modules inside the `tests/` directory. Run the test suite:
  ```bash
  pytest -q
  ```

---

## 6. Agent Communication & Command Formatting
* **Shell Preference:** When writing command-line examples in chat responses, always format them for Windows **CMD (Command Prompt)** instead of PowerShell.
  * Use caret (`^`) for line continuations.
  * Use backslashes (`\`) for file paths.

---

## 7. Pre-Push Safety Rule
* Never push code until the local repo passes the same quality gates used by CI:
  * `ruff check src main.py tools tests`
  * `pytest -q`
* If you add new modules or tests, format them first and fix every `ruff` error in the touched files before considering the change ready.
* If local Windows temp/cache permissions interfere with `pytest`, rerun with a workspace-local temp directory before assuming the test suite is broken.

---

## 8. Roboflow Workflows Integration

### 8.1 API Key & Authentication
* Never hardcode the Roboflow API key.
* Always read from the environment variable `ROBOFLOW_API_KEY` (e.g., `os.environ.get("ROBOFLOW_API_KEY")`).

### 8.2 Client Usage
* Use the official `inference-sdk` Python package to interface with the serverless endpoint:
```python
from inference_sdk import InferenceHTTPClient
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
```
* Wrap workflow executions with retries and exponential backoff using the `backoff` library to survive rate limits and transient connection issues.

### 8.3 Input/Output Data Contracts
* **Inputs:** Provide the `images` dictionary with the key `"image"`. The value can be a base64 encoded string, local path, or public URL.
* **Outputs:** The response returns a list of dictionaries. Each entry is keyed by:
  - `count_objects` (the count of objects detected).
  - `predictions` (a dictionary containing `image` metadata and `predictions` list of bounding box schemas).
  - `output_image` (a base64 encoded string representing the visualized output frame).
* **Defensive Parsing:** Bounding boxes from Roboflow are center-based (`x`, `y`, `width`, `height`). Always convert them to top-left / bottom-right coordinates `[x1, y1, x2, y2]` for consistency with the rest of the detection/tracking modules.
