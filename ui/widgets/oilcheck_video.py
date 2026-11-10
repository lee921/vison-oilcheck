"""监控画面控件：视频、ROI、检测框与状态横幅。

与旧预览控件不同，这里直接使用 QPainter 绘制，避免每帧做 OpenCV 叠加和
PIL 文本转换；ROI 使用归一化坐标，窗口缩放时同步缩放。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.oilcheck.roi import Roi
from app.oilcheck.types import BBox, Detection, TrackSnapshot

# 检测框调色板：前四项对应配置里的 hand / brush / oil_station / screw，
# 顺序由 monitor_page 决定；其余留给模型后续新增的类别。
PALETTE: tuple[str, ...] = (
    "#ff9f0a",
    "#ff375f",
    "#ffd60a",
    "#32d74b",
    "#bf5af2",
    "#64d2ff",
    "#5ac8fa",
    "#ff2d55",
)


@dataclass(frozen=True)
class RoiOverlay:
    name: str
    roi: Roi
    color: str
    dashed: bool = False


class OilCheckVideoView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._image: QImage | None = None
        self._source_size: tuple[int, int] = (0, 0)
        self._rois: tuple[RoiOverlay, ...] = ()
        self._detections: tuple[Detection, ...] = ()
        self._tracks: tuple[TrackSnapshot, ...] = ()
        self._show_track_ids = True
        self._banner = ""
        self._hint = "等待视频帧"
        self._label_colors: dict[str, str] = {}
        self._fallback_colors = PALETTE

    def set_frame(self, bgr: np.ndarray, ts: float) -> bool:
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return False
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return False
        if not bgr.flags["C_CONTIGUOUS"]:
            bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        image = QImage(
            bgr.data, w, h, int(bgr.strides[0]), QImage.Format.Format_BGR888
        ).copy()
        self._image = image
        self._source_size = (int(w), int(h))
        self._hint = ""
        self.update()
        return True

    def set_overlays(
        self,
        *,
        rois: tuple[RoiOverlay, ...],
        detections: tuple[Detection, ...],
        tracks: tuple[TrackSnapshot, ...] = (),
        banner: str = "",
        label_colors: dict[str, str] | None = None,
    ) -> None:
        """``label_colors`` 由调用方按配置里的类别名构造。

        未登记的类别按调色板顺延取色，这样模型新增类别也能画出来，
        不需要改 UI 代码。

        ``tracks`` 是管线在跟踪范围内的轨迹（手/刷头的视觉轨迹 ID）。
        是否标出 ID 由 :meth:`set_show_track_ids` 控制。
        """

        self._rois = rois
        self._detections = detections
        self._tracks = tracks
        self._banner = banner
        if label_colors:
            self._label_colors = dict(label_colors)
        self.update()

    def set_show_track_ids(self, show: bool) -> None:
        """是否在画面上标出跟踪 ID。"""

        if self._show_track_ids == show:
            return
        self._show_track_ids = bool(show)
        self.update()

    def clear(self, hint: str = "等待视频帧") -> None:
        self._image = None
        self._hint = hint
        self._banner = ""
        self._rois = ()
        self._detections = ()
        self._tracks = ()
        self.update()

    @property
    def has_frame(self) -> bool:
        return self._image is not None

    def paintEvent(self, event) -> None:  # noqa: N802
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08090a"))

        image_rect = self._image_rect()
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(image_rect, self._image)
            self._draw_rois(painter, image_rect)
            self._draw_detections(painter, image_rect)
            if self._show_track_ids:
                self._draw_track_ids(painter, image_rect)
        else:
            painter.setPen(QColor("#8a8f98"))
            painter.setFont(QFont("", 12))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._hint or "等待视频帧"
            )

        self._draw_banner(painter)
        painter.end()

    # ------------------------------------------------------------------
    def _image_rect(self) -> QRectF:
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        if self._image is None or self._source_size[0] <= 0:
            return QRectF(0, 0, widget_w, widget_h)

        src_w, src_h = self._source_size
        scale = min(widget_w / src_w, widget_h / src_h)
        draw_w = src_w * scale
        draw_h = src_h * scale
        return QRectF(
            (widget_w - draw_w) * 0.5,
            (widget_h - draw_h) * 0.5,
            draw_w,
            draw_h,
        )

    def _map_point(self, x: float, y: float, image_rect: QRectF) -> QPointF:
        src_w, src_h = self._source_size
        if src_w <= 0 or src_h <= 0:
            return QPointF(0, 0)
        return QPointF(
            image_rect.x() + x / src_w * image_rect.width(),
            image_rect.y() + y / src_h * image_rect.height(),
        )

    def _draw_rois(self, painter: QPainter, image_rect: QRectF) -> None:
        src_w, src_h = self._source_size
        for item in self._rois:
            color = QColor(item.color)
            pen = QPen(color)
            pen.setWidth(2)
            if item.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            points = item.roi.pixel_points(src_w, src_h)
            if item.roi.kind == "rect":
                (x1, y1), (x2, y2) = points
                painter.drawRect(
                    QRectF(
                        self._map_point(x1, y1, image_rect),
                        self._map_point(x2, y2, image_rect),
                    )
                )
                label_point = self._map_point(x1, y1, image_rect)
            else:
                polygon = QPolygonF(
                    [self._map_point(x, y, image_rect) for x, y in points]
                )
                painter.drawPolygon(polygon)
                label_point = polygon[0]

            if item.name:
                painter.setFont(QFont("", 9))
                label_rect = QRectF(
                    label_point.x() + 4,
                    label_point.y() - 18,
                    140,
                    16,
                )
                painter.fillRect(label_rect, QColor(0, 0, 0, 150))
                painter.setPen(color)
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    item.name,
                )

    def _color_for(self, label: str) -> str:
        """已配置的类别用配置颜色，未知类别按名字稳定散列取调色板色。"""

        color = self._label_colors.get(label)
        if color:
            return color
        index = sum(ord(ch) for ch in label) % len(self._fallback_colors)
        return self._fallback_colors[index]

    def _draw_detections(self, painter: QPainter, image_rect: QRectF) -> None:
        painter.setFont(QFont("", 9))
        for det in self._detections:
            color = QColor(self._color_for(det.label))

            top_left = self._map_point(det.box.x1, det.box.y1, image_rect)
            bottom_right = self._map_point(det.box.x2, det.box.y2, image_rect)
            box_rect = QRectF(top_left, bottom_right).normalized()

            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box_rect)

            if det.point is not None:
                point = self._map_point(det.point[0], det.point[1], image_rect)
                painter.setBrush(color)
                painter.drawEllipse(point, 4.0, 4.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)

            label = f"{det.label} {det.confidence:.2f}"
            text_rect = QRectF(
                box_rect.x(),
                max(image_rect.y(), box_rect.y() - 18),
                max(80.0, box_rect.width()),
                16,
            )
            painter.fillRect(
                text_rect, QColor(color.red(), color.green(), color.blue(), 190)
            )
            painter.setPen(QColor("#000000"))
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

    def _draw_track_ids(self, painter: QPainter, image_rect: QRectF) -> None:
        """在跟踪目标上标出 ID。

        手标 ``#100004``、刷头标 ``#200001``——前缀区分对象类型，后缀是
        跟踪器分配的视觉轨迹 ID。

        实测轨迹不再重复画框——检测框已经画过同一个目标，重复描边会形成重影；
        只在框外侧贴一个 ID 徽标。丢失中的轨迹没有检测框可依附，画虚线框
        并给 ID 加 ``?``，避免把预测位置当成实测。
        """

        painter.setFont(QFont("", 9))
        for track in self._tracks:
            color = QColor(self._color_for(track.label))
            box_rect = self._track_rect(track, image_rect)

            if not track.observed:
                pen = QPen(color)
                pen.setWidth(1)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(box_rect)

            text = self._track_label(track)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(text) + 10
            text_rect = self._track_badge_rect(
                box_rect, image_rect, text_w
            )
            # 深底彩色字：检测框标签是彩色底黑字，两者一眼可分——
            # 彩底黑字是“模型看到了什么”，深底彩字是“管线在跟踪谁”。
            painter.fillRect(text_rect, QColor(0, 0, 0, 190))
            painter.setPen(color)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignCenter,
                text,
            )

    @staticmethod
    def _track_badge_rect(
        box_rect: QRectF, image_rect: QRectF, text_w: float
    ) -> QRectF:
        """ID 徽标位置：优先贴在框右侧外面，贴到画面外时退回框内左上角。

        夹具里的丝杆是逐行紧排的窄目标，徽标贴在框下方会压住下一根的检测框；
        贴到右侧空白处既不遮挡、也不和左上角的类别标签打架。
        """

        height = 15.0
        x = box_rect.right() + 2.0
        if x + text_w > image_rect.right():
            x = max(image_rect.x(), box_rect.right() - text_w)
        y = box_rect.y()
        if y + height > image_rect.bottom():
            y = max(image_rect.y(), image_rect.bottom() - height)
        return QRectF(x, y, text_w, height)

    @staticmethod
    def _track_label(track: TrackSnapshot) -> str:
        text = f"#{track.track_id}"
        if not track.observed:
            text = f"{text} ?"
        return text

    def _track_rect(self, track: TrackSnapshot, image_rect: QRectF) -> QRectF:
        box: BBox = track.box
        return QRectF(
            self._map_point(box.x1, box.y1, image_rect),
            self._map_point(box.x2, box.y2, image_rect),
        ).normalized()

    def _draw_banner(self, painter: QPainter) -> None:
        if not self._banner:
            return
        painter.setFont(QFont("", 10, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(self._banner) + 20
        rect = QRectF(10, 10, text_w, 28)
        painter.fillRect(rect, QColor(0, 0, 0, 170))
        painter.setPen(QColor("#f2f2f7"))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            self._banner,
        )
