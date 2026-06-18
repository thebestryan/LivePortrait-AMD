# coding: utf-8

"""
Resolve the best available ONNX Runtime execution provider.

This keeps provider selection robust across NVIDIA CUDA, AMD ROCm and
CPU-only onnxruntime builds. On AMD GPUs the CUDA provider is not present,
so we transparently fall back to ROCm (when onnxruntime-rocm is installed)
or to CPU, avoiding noisy 'Specified provider ... is not available' warnings.

Note: this only affects the ONNX models (face detection + landmarks). The
main PyTorch pipeline already runs on AMD via ROCm, which exposes the CUDA
device API, so no changes are needed there.
"""

import onnxruntime


def resolve_onnx_providers(prefer='cuda', device_id=0):
    """Return a providers list for onnxruntime.InferenceSession.

    prefer: 'cuda'/'gpu' to request GPU acceleration, 'mps' for CoreML,
            anything else (e.g. 'cpu') for CPU.
    device_id: GPU index, forwarded to the CUDA/ROCm provider.
    """
    available = onnxruntime.get_available_providers()

    if prefer.lower() == 'mps' and 'CoreMLExecutionProvider' in available:
        return ['CoreMLExecutionProvider']

    if prefer.lower() in ('cuda', 'gpu', 'rocm'):
        if 'CUDAExecutionProvider' in available:
            return [('CUDAExecutionProvider', {'device_id': device_id})]
        if 'ROCMExecutionProvider' in available:
            return [('ROCMExecutionProvider', {'device_id': device_id})]

    return ['CPUExecutionProvider']
