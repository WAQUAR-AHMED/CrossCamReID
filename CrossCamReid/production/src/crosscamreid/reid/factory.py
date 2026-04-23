from __future__ import annotations

from .base import BaseReIDBackend
from .onnx_backend import ONNXReIDBackend
from .tensorrt_backend import TensorRTReIDBackend


def create_reid_backend(
    backend: str,
    onnx_path: str,
    tensorrt_engine_path: str | None,
) -> BaseReIDBackend:
    backend = backend.lower().strip()
    if backend == "onnxruntime":
        return ONNXReIDBackend(onnx_path)

    if backend == "tensorrt":
        if not tensorrt_engine_path:
            raise ValueError(
                "runtime.reid_backend=tensorrt requires models.reid_tensorrt_engine_path"
            )
        return TensorRTReIDBackend(tensorrt_engine_path)

    raise ValueError(f"Unsupported ReID backend: {backend}")

