from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .keypoints import (
    KP_LEFT_SHOULDER,
    KP_RIGHT_SHOULDER,
    REQUIRED_KPS,
)
from .processor import UNKNOWN_LABEL

_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 200, 50),
    (200, 50, 255),
    (50, 200, 255),
    (255, 120, 200),
    (120, 255, 150),
    (150, 120, 255),
    (255, 180, 100),
    (100, 255, 200),
    (180, 100, 255),
]
_UNKNOWN_COLOR = (110, 110, 110)
_REGION_COLOR = (255, 255, 0)


def _color_for(sid):
    if sid == UNKNOWN_LABEL:
        return _UNKNOWN_COLOR
    return _COLORS[(int(sid) - 1) % len(_COLORS)]


def draw_overlay(frame, records, fps, sid_count, cam_label: str, mode_label: str, config: AppConfig):
    for rec in records:
        x1, y1, x2, y2 = [int(c) for c in rec["bbox"]]
        sid = rec["sid"]
        tid = rec["tid"]
        score = rec["similarity_score"]
        kp_ok = rec["keypoint_valid"]
        color = _color_for(sid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        region = rec.get("region_bbox")
        if region is not None:
            rx1, ry1, rx2, ry2 = region
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), _REGION_COLOR, 2)
            cv2.putText(
                frame,
                "ReID region",
                (rx1, ry2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                _REGION_COLOR,
                1,
            )

        kp_xy = rec["kp_xy"]
        kp_conf = rec["kp_conf"]
        for idx in REQUIRED_KPS:
            if kp_conf[idx] >= config.gating.keypoint_conf_thresh:
                cv2.circle(frame, (int(kp_xy[idx, 0]), int(kp_xy[idx, 1])), 4, color, -1)
                cv2.circle(frame, (int(kp_xy[idx, 0]), int(kp_xy[idx, 1])), 5, (255, 255, 255), 1)

        for idx in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER):
            if kp_conf[idx] >= config.gating.keypoint_conf_thresh:
                cv2.circle(
                    frame, (int(kp_xy[idx, 0]), int(kp_xy[idx, 1])), 4, _REGION_COLOR, -1
                )

        sid_txt = "UNKNOWN" if sid == UNKNOWN_LABEL else f"SID {sid}"
        score_txt = f" {score:.2f}" if score is not None else ""
        kp_tag = "KP+" if kp_ok else "KP-"
        phase = ""
        if rec["enroll_left"] > 0:
            done = config.enrollment.enroll_frames - rec["enroll_left"]
            phase = f" | ENROLL {done}/{config.enrollment.enroll_frames}"
        elif sid == UNKNOWN_LABEL and kp_ok and rec["qualified"] > 0:
            phase = f" | QUAL {rec['qualified']}/{config.enrollment.qualify_frames}"
        label = f"TID {tid} | {sid_txt}{score_txt} | {kp_tag}{phase}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    header = f"{cam_label} [{mode_label}] (torso-region ReID)"
    cv2.putText(
        frame,
        header,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255) if mode_label == "MASTER" else (255, 200, 0),
        2,
    )

    overlay = [
        f"FPS:        {fps:5.1f}",
        f"Detections: {len(records)}",
        f"Total SIDs: {sid_count}",
        f"ReID mode:   {config.runtime.reid_backend}",
    ]
    y = 60
    for line in overlay:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 24
    return frame


def combine_side_by_side(f1, f2, total_width: int):
    half_w = total_width // 2

    def _fit(frame, tag):
        if frame is None:
            blank = np.zeros((540, half_w, 3), dtype=np.uint8)
            cv2.putText(blank, f"{tag}: No Signal", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return blank
        h, w = frame.shape[:2]
        return cv2.resize(frame, (half_w, int(h * half_w / w)))

    a = _fit(f1, "CAM 1")
    b = _fit(f2, "CAM 2")
    if a.shape[0] != b.shape[0]:
        h = max(a.shape[0], b.shape[0])
        a = cv2.resize(a, (half_w, h))
        b = cv2.resize(b, (half_w, h))
    return np.hstack([a, b])

