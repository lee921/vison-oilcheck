"""检测服务：配置文件被外部改动后自动重载。

现场会直接编辑 ``<数据目录>/config/oilcheck.json`` 改 ROI。程序原本只在
启动时读一次（或从 UI 保存时 reload），手工改文件不生效、也很难看出原因。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.oilcheck.config import (
    OilCheckConfigService,
    add_json_comments,
    load_json_with_comments,
)
from app.oilcheck.service import OilCheckService


class _FakeManager:
    def get_session(self, camera_id):  # noqa: ANN001, ARG002
        return None

    def cameras(self):
        return ()


class _FakeEventStore:
    pass


def _make_service(tmp_path: Path) -> tuple[OilCheckService, Path]:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "oilcheck.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.5",
                "pipeline": {"max_fps": 15},
                "stations": [
                    {
                        "station_id": "ST01",
                        "name": "1号取料工位",
                        "camera_id": "cam",
                        "enabled": True,
                        "tray_roi": {
                            "kind": "rect",
                            "points": [[0.5, 0.1], [0.7, 0.4]],
                        },
                        "work_roi": {
                            "kind": "rect",
                            "points": [[0.3, 0.0], [0.9, 0.9]],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service = OilCheckConfigService(tmp_path)
    cfg = service.load()
    oil_service = OilCheckService(
        manager=_FakeManager(),
        config_service=service,
        config=cfg,
        event_store=_FakeEventStore(),
        data_root=tmp_path,
    )
    return oil_service, path


def _edit(path: Path, mutate) -> None:
    """在保留注释的前提下改配置，并把 mtime 推后。"""

    data = load_json_with_comments(path)
    mutate(data)
    # 明确推后 mtime：同一秒内的两次写入在粗粒度文件系统上 mtime 相同。
    path.write_text(add_json_comments(data, {}) + "\n", encoding="utf-8")
    current = path.stat().st_mtime
    path.touch()
    assert path.stat().st_mtime >= current


def test_roi_edit_on_disk_is_picked_up(tmp_path):
    """手改 tray_roi 后，服务的判定区随之更新。"""

    service, path = _make_service(tmp_path)
    assert service.station("ST01").tray_roi.points == ((0.5, 0.1), (0.7, 0.4))

    new_points = [[0.11, 0.22], [0.33, 0.44]]
    _edit(
        path,
        lambda data: data["stations"][0]["tray_roi"].update(points=new_points),
    )
    service._reload_if_file_changed()

    assert service.station("ST01").tray_roi.points == ((0.11, 0.22), (0.33, 0.44))


def test_pipeline_edit_on_disk_is_picked_up(tmp_path):
    service, path = _make_service(tmp_path)
    assert service.config.pipeline.max_fps == 15

    _edit(path, lambda data: data["pipeline"].update(max_fps=8))
    service._reload_if_file_changed()

    assert service.config.pipeline.max_fps == 8


def test_touch_without_content_change_does_not_reload(tmp_path):
    """只改 mtime（备份工具、touch）不重建运行时。"""

    service, path = _make_service(tmp_path)
    runtimes = service._runtimes
    path.touch()
    service._reload_if_file_changed()
    assert service._runtimes is runtimes, "内容没变就不该重建运行时"


def test_broken_file_keeps_previous_config(tmp_path):
    """文件被改成非法 JSON 时保留旧配置，不能把服务搞崩。"""

    service, path = _make_service(tmp_path)
    service._reload_if_file_changed()  # 建立基线

    path.write_text("{ 这不是 JSON", encoding="utf-8")
    service._reload_if_file_changed()

    assert service.config.pipeline.max_fps == 15
    assert service.station("ST01") is not None


def test_repeated_checks_do_not_loop(tmp_path):
    """重载后 mtime 基线要更新，否则每帧都会重载一次。"""

    service, path = _make_service(tmp_path)
    _edit(path, lambda data: data["pipeline"].update(max_fps=9))
    service._reload_if_file_changed()
    baseline = service._config_mtime

    service._reload_if_file_changed()
    service._reload_if_file_changed()
    assert service._config_mtime == baseline
    assert service.config.pipeline.max_fps == 9
