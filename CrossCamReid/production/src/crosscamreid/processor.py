from __future__ import annotations

from .config import AppConfig
from .keypoints import keypoint_gate, torso_region_bbox
from .reid.base import BaseReIDBackend
from .state import TIDStateManager
from .store import SIDStore

UNKNOWN_LABEL = "UNKNOWN"


def _blank_record(tid: int, bbox, kp_xy, kp_conf, kp_ok: bool) -> dict:
    return {
        "tid": int(tid),
        "sid": UNKNOWN_LABEL,
        "keypoint_valid": bool(kp_ok),
        "similarity_score": None,
        "bbox": bbox,
        "kp_xy": kp_xy,
        "kp_conf": kp_conf,
        "qualified": 0,
        "enroll_left": 0,
        "region_bbox": None,
    }


def process_master(
    frame,
    bbox,
    kp_xy,
    kp_conf,
    tid,
    reid: BaseReIDBackend,
    store: SIDStore,
    states: TIDStateManager,
    config: AppConfig,
) -> dict:
    tid = int(tid)
    kp_ok = keypoint_gate(kp_conf, config.gating)
    record = _blank_record(tid, bbox, kp_xy, kp_conf, kp_ok)

    if not kp_ok:
        return record

    state = states.get(tid)
    record["qualified"] = state.qualified
    record["enroll_left"] = state.enroll_left

    region = torso_region_bbox(kp_xy, kp_conf, frame.shape, config.gating)
    record["region_bbox"] = region

    if state.locked_sid is not None:
        record["sid"] = state.locked_sid
        return record

    if region is None:
        return record

    embedding = reid.embed(frame, region)
    if embedding is None:
        return record

    if state.enroll_left > 0 and state.new_sid is not None:
        store.append(state.new_sid, embedding)
        state.enroll_left -= 1
        record["sid"] = state.new_sid
        record["enroll_left"] = state.enroll_left
        if state.enroll_left == 0:
            state.locked_sid = state.new_sid
            state.new_sid = None
        return record

    matched_sid, score = store.search_top1(embedding)
    if matched_sid is not None and score >= config.gating.match_thresh:
        state.locked_sid = matched_sid
        record["sid"] = matched_sid
        record["similarity_score"] = float(score)
        return record

    state.qualified += 1
    record["qualified"] = state.qualified
    record["similarity_score"] = float(score) if score > 0 else None

    if state.qualified >= config.enrollment.qualify_frames:
        new_sid = store.new_sid(embedding)
        state.new_sid = new_sid
        state.enroll_left = config.enrollment.enroll_frames - 1
        record["sid"] = new_sid
        record["enroll_left"] = state.enroll_left
        record["similarity_score"] = None
        if state.enroll_left == 0:
            state.locked_sid = new_sid
            state.new_sid = None

    return record


def process_slave(
    frame,
    bbox,
    kp_xy,
    kp_conf,
    tid,
    reid: BaseReIDBackend,
    store: SIDStore,
    states: TIDStateManager,
    config: AppConfig,
) -> dict:
    tid = int(tid)
    kp_ok = keypoint_gate(kp_conf, config.gating)
    record = _blank_record(tid, bbox, kp_xy, kp_conf, kp_ok)

    if not kp_ok:
        return record

    region = torso_region_bbox(kp_xy, kp_conf, frame.shape, config.gating)
    record["region_bbox"] = region

    state = states.get(tid)
    if state.locked_sid is not None:
        record["sid"] = state.locked_sid
        return record

    if region is None:
        return record

    embedding = reid.embed(frame, region)
    if embedding is None:
        return record

    matched_sid, score = store.search_top1(embedding)
    record["similarity_score"] = float(score) if score > 0 else None

    if matched_sid is not None and score >= config.gating.match_thresh:
        state.locked_sid = matched_sid
        record["sid"] = matched_sid

    return record

