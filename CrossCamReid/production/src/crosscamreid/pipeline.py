from __future__ import annotations

import json
import time

import cv2
from ultralytics import YOLO

from .capture import RTSPCapture
from .config import AppConfig
from .overlay import combine_side_by_side, draw_overlay
from .processor import process_master, process_slave
from .reid.factory import create_reid_backend
from .state import TIDStateManager
from .store import SIDStore


def _run_stream(
    pose: YOLO,
    frame,
    config: AppConfig,
    processor,
    reid_backend,
    store,
    states,
):
    result = pose.track(
        frame,
        persist=True,
        classes=[0],
        conf=config.gating.person_conf_thresh,
        tracker=config.runtime.tracker,
        verbose=False,
    )[0]

    records: list[dict] = []
    alive: set[int] = set()

    if (
        result.boxes is not None
        and len(result.boxes) > 0
        and result.keypoints is not None
        and result.boxes.id is not None
    ):
        boxes = result.boxes.xyxy.cpu().numpy()
        tids = result.boxes.id.cpu().numpy().astype(int)
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):
            bbox = [float(c) for c in boxes[i]]
            rec = processor(
                frame,
                bbox,
                kp_xy[i],
                kp_conf[i],
                int(tids[i]),
                reid_backend,
                store,
                states,
                config,
            )
            records.append(rec)
            alive.add(int(tids[i]))

    states.forget(alive)
    return records


def run_app(config: AppConfig) -> int:
    print("\n" + "=" * 70)
    print("  CrossCamReid (Dual-camera, torso-region ReID)")
    print(f"  CAM 1 MASTER     : {config.sources.master}")
    print(f"  CAM 2 SLAVE      : {config.sources.slave}")
    print(f"  Pose model       : {config.models.pose_path}")
    print(f"  ReID backend     : {config.runtime.reid_backend}")
    print(f"  ReID ONNX model  : {config.models.reid_onnx_path}")
    print(f"  ReID TRT engine  : {config.models.reid_tensorrt_engine_path}")
    print(f"  DB path          : {config.database.path}")
    print(f"  Collection       : {config.database.collection}")
    print(f"  Match threshold  : {config.gating.match_thresh}")
    print(f"  Region pad frac  : {config.gating.region_pad_frac}")
    print(f"  Fresh DB         : {not config.database.keep_db}")
    print("=" * 70 + "\n")

    print("[Pose] Loading master and slave pose models...")
    pose_master = YOLO(config.models.pose_path)
    pose_slave = YOLO(config.models.pose_path)

    reid_backend = create_reid_backend(
        backend=config.runtime.reid_backend,
        onnx_path=config.models.reid_onnx_path,
        tensorrt_engine_path=config.models.reid_tensorrt_engine_path,
    )
    store = SIDStore(
        db_path=config.database.path,
        collection=config.database.collection,
        dim=reid_backend.dim,
        fresh=not config.database.keep_db,
        max_embeddings_per_sid=config.gating.max_embeddings_per_sid,
    )

    states_master = TIDStateManager()
    states_slave = TIDStateManager()

    cap1 = RTSPCapture(config.sources.master, "Cam1-M", config.capture).start()
    cap2 = RTSPCapture(config.sources.slave, "Cam2-S", config.capture).start()

    fps_count, fps_start, fps = 0, time.time(), 0.0
    print("[Main] Running. Press Q to quit.")

    try:
        while True:
            frame1 = cap1.get_frame()
            frame2 = cap2.get_frame()
            if frame1 is None and frame2 is None:
                time.sleep(0.05)
                continue

            records1: list[dict] = []
            records2: list[dict] = []

            if frame1 is not None:
                records1 = _run_stream(
                    pose_master,
                    frame1,
                    config,
                    process_master,
                    reid_backend,
                    store,
                    states_master,
                )
            if frame2 is not None:
                records2 = _run_stream(
                    pose_slave,
                    frame2,
                    config,
                    process_slave,
                    reid_backend,
                    store,
                    states_slave,
                )

            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count, fps_start = 0, time.time()

            if config.runtime.log_json and (records1 or records2):
                payload = {
                    "fps": round(fps, 2),
                    "cam1_master": [
                        {k: v for k, v in rec.items() if k not in ("kp_xy", "kp_conf")} for rec in records1
                    ],
                    "cam2_slave": [
                        {k: v for k, v in rec.items() if k not in ("kp_xy", "kp_conf")} for rec in records2
                    ],
                }
                print(json.dumps(payload))

            if config.runtime.no_display:
                if records1 or records2:
                    def _fmt(tag, recs):
                        parts = []
                        for rec in recs:
                            score = rec["similarity_score"]
                            suffix = f"({score:.2f})" if score is not None else ""
                            parts.append(f"{tag}T{rec['tid']}->{rec['sid']}{suffix}")
                        return ", ".join(parts)

                    line = " | ".join(
                        x for x in (_fmt("M:", records1), _fmt("S:", records2)) if x
                    )
                    print(f"[{fps:5.1f} fps] {line}")
            else:
                if frame1 is not None:
                    frame1 = draw_overlay(
                        frame1,
                        records1,
                        fps,
                        store.total_sids(),
                        "CAM 1",
                        "MASTER",
                        config,
                    )
                if frame2 is not None:
                    frame2 = draw_overlay(
                        frame2,
                        records2,
                        fps,
                        store.total_sids(),
                        "CAM 2",
                        "SLAVE",
                        config,
                    )
                combined = combine_side_by_side(frame1, frame2, config.runtime.display_width)
                cv2.imshow("CrossCamReid | Q to quit", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        cap1.stop()
        cap2.stop()
        if not config.runtime.no_display:
            cv2.destroyAllWindows()

    print(f"[Main] Done. Total SIDs: {store.total_sids()}")
    return 0

