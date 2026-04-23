from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SourceConfig:
    master: str
    slave: str


@dataclass
class ModelConfig:
    pose_path: str
    reid_onnx_path: str
    reid_tensorrt_engine_path: str | None


@dataclass
class CaptureConfig:
    buffer_size: int
    reconnect_initial_delay_sec: float
    reconnect_max_delay_sec: float
    max_read_failures: int


@dataclass
class GatingConfig:
    person_conf_thresh: float
    keypoint_conf_thresh: float
    match_thresh: float
    min_region_side: int
    region_pad_frac: float
    max_embeddings_per_sid: int


@dataclass
class EnrollmentConfig:
    qualify_frames: int
    enroll_frames: int


@dataclass
class DatabaseConfig:
    path: str
    collection: str
    keep_db: bool


@dataclass
class RuntimeConfig:
    tracker: str
    reid_backend: str
    no_display: bool
    display_width: int
    log_json: bool


@dataclass
class AppConfig:
    sources: SourceConfig
    models: ModelConfig
    capture: CaptureConfig
    gating: GatingConfig
    enrollment: EnrollmentConfig
    database: DatabaseConfig
    runtime: RuntimeConfig


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required config key: {key}")
    return raw[key]


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_config(config_path: str) -> AppConfig:
    cfg_path = Path(config_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")

    base_dir = cfg_path.parent

    sources = _require(raw, "sources")
    models = _require(raw, "models")
    capture = _require(raw, "capture")
    gating = _require(raw, "gating")
    enrollment = _require(raw, "enrollment")
    database = _require(raw, "database")
    runtime = _require(raw, "runtime")

    app_cfg = AppConfig(
        sources=SourceConfig(
            master=str(_require(sources, "master")),
            slave=str(_require(sources, "slave")),
        ),
        models=ModelConfig(
            pose_path=str(_resolve_path(base_dir, str(_require(models, "pose_path")))),
            reid_onnx_path=str(_resolve_path(base_dir, str(_require(models, "reid_onnx_path")))),
            reid_tensorrt_engine_path=_resolve_path(
                base_dir, models.get("reid_tensorrt_engine_path")
            ),
        ),
        capture=CaptureConfig(
            buffer_size=int(_require(capture, "buffer_size")),
            reconnect_initial_delay_sec=float(_require(capture, "reconnect_initial_delay_sec")),
            reconnect_max_delay_sec=float(_require(capture, "reconnect_max_delay_sec")),
            max_read_failures=int(_require(capture, "max_read_failures")),
        ),
        gating=GatingConfig(
            person_conf_thresh=float(_require(gating, "person_conf_thresh")),
            keypoint_conf_thresh=float(_require(gating, "keypoint_conf_thresh")),
            match_thresh=float(_require(gating, "match_thresh")),
            min_region_side=int(_require(gating, "min_region_side")),
            region_pad_frac=float(_require(gating, "region_pad_frac")),
            max_embeddings_per_sid=int(_require(gating, "max_embeddings_per_sid")),
        ),
        enrollment=EnrollmentConfig(
            qualify_frames=int(_require(enrollment, "qualify_frames")),
            enroll_frames=int(_require(enrollment, "enroll_frames")),
        ),
        database=DatabaseConfig(
            path=str(_resolve_path(base_dir, str(_require(database, "path")))),
            collection=str(_require(database, "collection")),
            keep_db=bool(_require(database, "keep_db")),
        ),
        runtime=RuntimeConfig(
            tracker=str(_require(runtime, "tracker")),
            reid_backend=str(_require(runtime, "reid_backend")).lower().strip(),
            no_display=bool(_require(runtime, "no_display")),
            display_width=int(_require(runtime, "display_width")),
            log_json=bool(_require(runtime, "log_json")),
        ),
    )

    if app_cfg.runtime.reid_backend not in {"onnxruntime", "tensorrt"}:
        raise ValueError("runtime.reid_backend must be one of: onnxruntime, tensorrt")

    if app_cfg.enrollment.enroll_frames < 1:
        raise ValueError("enrollment.enroll_frames must be >= 1")

    if app_cfg.enrollment.qualify_frames < 1:
        raise ValueError("enrollment.qualify_frames must be >= 1")

    return app_cfg

