from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.oilcheck.recorder import TrainingRecorder, list_segments


def test_training_recorder_writes_video_and_metadata(tmp_path):
    recorder = TrainingRecorder(
        camera_id="cam_test",
        camera_name="测试相机",
        root=tmp_path,
        fps=5,
        segment_seconds=10.0,
        codec="mp4v",
        station_meta={
            "station_id": "ST01",
            "station_name": "测试工位",
            "rois": {"tray_roi": {"kind": "rect", "points": [[0, 0], [1, 1]]}},
        },
    )
    assert recorder.start()

    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    for i in range(10):
        frame[:] = (i * 10) % 255
        recorder.submit(1000.0 + i * 0.2, frame)
    recorder.stop()

    status = recorder.status()
    videos = list(tmp_path.glob("**/*.mp4"))
    if status.last_error and "无法创建视频文件" in status.last_error:
        pytest.skip(f"当前 OpenCV 无法编码 mp4v: {status.last_error}")

    assert videos, "应生成至少一个视频分段"
    assert status.frames_written > 0
    metadata = [p for p in tmp_path.glob("**/*.json") if p.name != "manifest.jsonl"]
    assert metadata
    data = json.loads(metadata[0].read_text(encoding="utf-8"))
    assert data["type"] == "training_video"
    assert data["status"] == "finished"
    assert data["station_id"] == "ST01"
    assert data["fps"] == 5
    assert data["width"] == 96
    assert data["height"] == 64

    segments = list_segments(tmp_path)
    assert segments
    assert Path(segments[0]["path"]).exists()


def test_recorder_drops_when_not_active(tmp_path):
    recorder = TrainingRecorder(
        camera_id="cam_test",
        camera_name="测试相机",
        root=tmp_path,
        fps=5,
        segment_seconds=10.0,
        codec="mp4v",
    )
    recorder.submit(1.0, np.zeros((8, 8, 3), dtype=np.uint8))
    assert recorder.status().frames_written == 0


def test_delete_segments_removes_video_metadata_and_manifest(tmp_path):
    from app.oilcheck.recorder import delete_segments

    day = tmp_path / "cam_test" / "2026-10-22"
    day.mkdir(parents=True)
    first = day / "first.mp4"
    second = day / "second.mp4"
    first.write_bytes(b"video-1")
    first.with_suffix(".json").write_text('{"type": "training_video"}')
    second.write_bytes(b"video-2")
    second.with_suffix(".json").write_text('{"type": "training_video"}')
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"path": str(first)}),
                json.dumps({"path": str(second)}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    deleted, errors = delete_segments(
        tmp_path,
        [str(first), str(second)],
        active_paths=[str(second)],
    )

    assert deleted == 1
    assert errors and "正在录制" in errors[0]
    assert not first.exists()
    assert not first.with_suffix(".json").exists()
    assert second.exists()
    assert second.with_suffix(".json").exists()

    lines = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [Path(item["path"]).name for item in lines] == ["second.mp4"]
