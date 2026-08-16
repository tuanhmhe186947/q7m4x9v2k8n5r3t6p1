import json

import torch


payload = {
    "visible_gpu_count": int(torch.cuda.device_count()),
    "cuda_available": bool(torch.cuda.is_available()),
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "gpu_model": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
}
print(json.dumps(payload, indent=2))
