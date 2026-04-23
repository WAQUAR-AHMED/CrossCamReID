from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BaseReIDBackend(ABC):
    input_hw = (224, 224)

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    def _preprocess(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x1, fw - 1))
        y1 = max(0, min(y1, fh - 1))
        x2 = max(x1 + 1, min(x2, fw))
        y2 = max(y1 + 1, min(y2, fh))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
            return None

        h, w = self.input_hw
        crop = cv2.resize(crop, (w, h))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - _IMAGENET_MEAN) / _IMAGENET_STD
        return crop.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def _postprocess(self, feat: np.ndarray) -> np.ndarray | None:
        feat = feat.flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm < 1e-6:
            return None
        return feat / norm

    @abstractmethod
    def embed(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        raise NotImplementedError

