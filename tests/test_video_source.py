from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.oilcheck.config import DemoConfig, DetectorConfig
from app.oilcheck.detector import _resolve_imgsz
from app.oilcheck.video_source import VideoFileSource


class _Args:
    def __init__(self, imgsz) -> None:
        self.imgsz = imgsz


class _Inner:
    def __init__(self, imgsz) -> None:
        self.args = {"imgsz": imgsz}


class _Model:
    def __init__(self, imgsz) -> None:
        self.model = _Inner(imgsz)


# ----------------------------------------------------------------------
# imgsz 自适应
# ----------------------------------------------------------------------
def test_imgsz_uses_training_size_from_checkpoint():
    assert _resolve_imgsz(_Model(1024), 0) == 1024


def test_imgsz_explicit_config_wins():
    assert _resolve_imgsz(_Model(1024), 640) == 640


def test_imgsz_falls_back_when_missing():
    assert _resolve_imgsz(_Model(None), 0) == 640
    assert _resolve_imgsz(object(), 0) == 640


def test_imgsz_rounds_up_to_stride():
    assert _resolve_imgsz(_Model(1000), 0) == 1024
    assert _resolve_imgsz(_Model(641), 0) == 672


def test_imgsz_accepts_sequence_training_size():
    assert _resolve_imgsz(_Model([1024, 1280]), 0) == 1280


def test_imgsz_accepts_attr_style_args():
    class _AttrModel:
        model = type("M", (), {"args": _Args(960)})()

    assert _resolve_imgsz(_AttrModel(), 0) == 960


# ----------------------------------------------------------------------
# 检测器配置
# ----------------------------------------------------------------------
def test_detector_config_matches_model_classes():
    cfg = DetectorConfig.from_dict({})
    assert cfg.hand_label == "hand"
    assert cfg.brush_label == "brush"
    assert cfg.tray_label == "oil_station"
    assert cfg.screw_label == "screw"
    assert cfg.imgsz == 0
    assert cfg.raw_mode is False


def test_detector_config_migrates_legacy_label_keys():
    cfg = DetectorConfig.from_dict({"brush_head_label": "old_brush"})
    assert cfg.brush_label == "old_brush"


def test_detector_config_rejects_removed_simulated_kind():
    assert DetectorConfig.from_dict({"kind": "simulated"}).kind == "null"


# ----------------------------------------------------------------------
# 示例视频配置
# ----------------------------------------------------------------------
def test_demo_config_parses_videos():
    cfg = DemoConfig.from_dict(
        {
            "loop": False,
            "fps_limit": 10,
            "videos": [
                {"name": "A", "path": "data/sample_video/a.mp4"},
                "data/sample_video/b.mp4",
            ],
        }
    )
    assert cfg.loop is False
    assert cfg.fps_limit == 10
    assert [v.name for v in cfg.videos] == ["A", "b"]
    assert cfg.video("b").path == "data/sample_video/b.mp4"
    assert cfg.video("missing") is None


# ----------------------------------------------------------------------
# 视频帧源
# ----------------------------------------------------------------------
def _make_video(path: Path, fps: int = 10, frames: int = 20) -> bool:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (64, 48)
    )
    if not writer.isOpened():
        return False
    for index in range(frames):
        img = np.full((48, 64, 3), index * 10 % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return True


def test_video_source_streams_frames_and_loops(tmp_path):
    path = tmp_path / "clip.mp4"
    if not _make_video(path):
        pytest.skip("当前环境无法写出测试视频")

    source = VideoFileSource(path, loop=True, fps_limit=30)
    source.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline and source.get() is None:
            time.sleep(0.02)
        first = source.get()
        assert first is not None
        assert (first.width, first.height) == (64, 48)

        # 循环播放：超过视频总时长后仍能拿到新帧。
        time.sleep(1.2)
        later = source.get()
        assert later is not None
        assert later.ts > first.ts

        status = source.status()
        assert status["playing"] is True
        assert status["native_fps"] == pytest.approx(10.0, rel=0.2)
    finally:
        source.stop()

    assert source.playing is False
    assert source.get() is None


def test_video_source_rejects_missing_file(tmp_path):
    source = VideoFileSource(tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError):
        source.start()


# ----------------------------------------------------------------------
# 设备选择
# ----------------------------------------------------------------------
def test_resolve_device_explicit_values_pass_through():
    from app.oilcheck.detector import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:1") == "cuda:1"
    assert _resolve_device("0") == "0"


def test_resolve_device_auto_falls_back_to_cpu_without_cuda():
    from app.oilcheck.detector import _resolve_device

    device = _resolve_device("auto")
    assert device in {"cpu", "cuda:0"}
    assert _resolve_device("") == device
