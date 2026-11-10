"""ROI 标定对话框：用示例视频帧代替相机画面标定。"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.oilcheck.config import OilCheckConfig  # noqa: E402

pytest.importorskip("PySide6.QtWidgets")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

SAMPLE_VIDEO = "data/sample_video/cam_6ca47a_20261027_105708.mp4"


def _make_video(path, fps=10, frames=20):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (64, 48)
    )
    if not writer.isOpened():
        return False
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 10 % 255, dtype=np.uint8))
    writer.release()
    return True


def _config(video_path: str):
    return OilCheckConfig.from_dict(
        {
            "demo": {"videos": [{"name": "标定用示例", "path": video_path}]},
            "stations": [
                {
                    "station_id": "ST01",
                    "name": "1号取料工位",
                    "camera_id": "cam_6ca47a",
                    "enabled": True,
                    "tray_roi": {"kind": "rect", "points": [[0.5, 0.1], [0.72, 0.44]]},
                    "work_roi": {"kind": "rect", "points": [[0.3, 0.0], [0.99, 0.95]]},
                    "oiling_roi": {
                        "kind": "rect",
                        "points": [[0.3, 0.0], [0.99, 0.95]],
                    },
                }
            ],
        }
    )


class _NoSessionManager:
    """相机不可用：验证视频标定完全不依赖相机。"""

    def get_session(self, _cam_id):
        return None


class _RecordingService:
    def __init__(self):
        self.saved = []

    def save(self, cfg):
        self.saved.append(cfg)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _dialog(cfg):
    from ui.widgets.roi_calibration import RoiCalibrationDialog

    service = _RecordingService()
    results = []
    dialog = RoiCalibrationDialog(
        manager=_NoSessionManager(),
        cfg_service=service,
        cfg=cfg,
        on_saved=results.append,
    )
    return dialog, service, results


def _flat(points):
    return tuple(value for point in points for value in point)


def _drag(canvas, mode, start, end):
    from PySide6.QtCore import QPointF

    rect = canvas._image_rect()
    pos = lambda p: QPointF(  # noqa: E731
        rect.x() + rect.width() * p[0], rect.y() + rect.height() * p[1]
    )

    class _Event:
        def position(self):
            return pos(end)

        def button(self):
            from PySide6.QtCore import Qt

            return Qt.MouseButton.LeftButton

    canvas._mode = mode
    canvas._drag_start = pos(start)
    canvas.mouseReleaseEvent(_Event())


@pytest.mark.skipif(not os.path.exists(SAMPLE_VIDEO), reason="缺少示例视频")
def test_roi_calibration_uses_sample_video_frame(qapp):
    dialog, service, results = _dialog(_config(SAMPLE_VIDEO))
    dialog.show()
    try:
        index = dialog._source_combo.findData("标定用示例")
        assert index > 0
        dialog._source_combo.setCurrentIndex(index)

        # 示例视频帧已进入画布，且不依赖相机。
        assert dialog._video_name == "标定用示例"
        assert dialog._canvas.source_size == (2560, 1440)
        assert dialog._canvas._editable is True
        assert dialog._btn_capture.isEnabled() is False

        # 拖动位置条应换帧：总帧数 846 的中点约在第 423 帧。
        dialog._video_slider.setValue(500)
        dialog._apply_video_seek()
        frame = dialog._video_pos.text().split("/")[0]
        assert frame.startswith("第 ")
        assert 400 <= int(frame.removeprefix("第 ")) <= 430
    finally:
        dialog.close()


def test_roi_calibration_seek_and_save_from_video(qapp, tmp_path):
    path = tmp_path / "clip.mp4"
    if not _make_video(path):
        pytest.skip("当前环境无法写出测试视频")

    dialog, service, results = _dialog(_config(str(path)))
    dialog.show()
    try:
        dialog._source_combo.setCurrentIndex(
            dialog._source_combo.findData("标定用示例")
        )
        assert dialog._canvas.source_size == (64, 48)

        dialog._video_slider.setValue(1000)
        dialog._apply_video_seek()
        assert dialog._video_pos.text().startswith("第 20/20 帧")

        _drag(dialog._canvas, "tray", (0.55, 0.15), (0.75, 0.40))
        _drag(dialog._canvas, "work", (0.30, 0.00), (0.99, 0.95))
        dialog._save()

        assert dialog.result() == 1
        assert len(service.saved) == 1 and len(results) == 1
        station = service.saved[0].stations[0]
        assert _flat(station.tray_roi.points) == pytest.approx(
            (0.55, 0.15, 0.75, 0.4), abs=1e-6
        )
        assert _flat(station.work_roi.points) == pytest.approx(
            (0.3, 0.0, 0.99, 0.95), abs=1e-6
        )
    finally:
        dialog.close()


def test_roi_calibration_rejects_missing_video(qapp, tmp_path):
    cfg = _config(str(tmp_path / "missing.mp4"))
    dialog, service, _results = _dialog(cfg)
    dialog.show()
    try:
        dialog._source_combo.setCurrentIndex(
            dialog._source_combo.findData("标定用示例")
        )
        # 视频打不开时回退到实时相机，不进入视频标定模式。
        assert dialog._video_name is None
        assert dialog._source_combo.currentIndex() == 0
        assert dialog._btn_capture.isEnabled() is True
        assert dialog._video_row.isHidden() is True
        assert service.saved == []
    finally:
        dialog.close()


def test_roi_calibration_hides_source_row_without_videos(qapp):
    cfg = _config("unused.mp4")
    cfg = OilCheckConfig.from_dict({**cfg.to_dict(), "demo": {"videos": []}})
    dialog, _service, _results = _dialog(cfg)
    try:
        assert dialog._source_combo.count() == 1
        assert all(w.isHidden() for w in dialog._source_widgets)
    finally:
        dialog.close()
