#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from PyQt5.QtCore import QPointF, QRect, QRectF, QSize, Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "当前环境没有安装 PyQt5。请先执行 `pip install PyQt5`，然后再启动桌面界面。"
    ) from exc

try:
    import brushnet_app_core as core
except Exception as exc:
    message = str(exc)
    if "huggingface-hub" in message and "transformers" in message:
        raise SystemExit(
            "BrushNet 环境依赖冲突：当前 `transformers` 和 `huggingface-hub` 版本不兼容。\n"
            "请在 `(brushnet)` 环境里执行下面这条命令后再启动：\n"
            'pip install "huggingface-hub>=0.34,<1.0" "transformers>=4.25.1,<5" -U\n'
            "如果你之前装过 1.x 版本的 huggingface-hub，建议先执行：\n"
            "pip uninstall -y huggingface-hub\n"
            '然后再执行上面的安装命令。'
        ) from exc
    raise


APP_STYLE = """
QMainWindow {
  background: #f5f1e8;
}

QWidget {
  font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
  font-size: 14px;
  color: #21352e;
}

QLabel#TitleLabel {
  font-size: 22px;
  font-weight: 700;
  color: #173f38;
}

QLabel#SubTitleLabel {
  font-size: 12px;
  color: #4b5f58;
}

QLabel#BrandLogoLabel {
  background: transparent;
  border: none;
  padding: 0;
}

QFrame#HeroCard,
QGroupBox {
  background: #fffdf8;
  border: 1px solid #d7ddd7;
  border-radius: 20px;
}

QGroupBox {
  font-weight: 700;
  margin-top: 14px;
  padding-top: 12px;
}

QGroupBox::title {
  subcontrol-origin: margin;
  left: 16px;
  padding: 0 6px;
  color: #173f38;
}

QPushButton {
  min-height: 36px;
  border-radius: 12px;
  background: #dfeee8;
  border: 1px solid #bdd3c8;
  padding: 0 14px;
  font-weight: 600;
}

QPushButton:hover {
  background: #d2e8df;
}

QPushButton#PrimaryButton {
  background: #0c7b6c;
  color: white;
  border: none;
}

QPushButton#PrimaryButton:hover {
  background: #0a6f61;
}

QPushButton#AccentButton {
  background: #d78745;
  color: white;
  border: none;
}

QPushButton#AccentButton:hover {
  background: #c87635;
}

QLineEdit,
QTextEdit,
QPlainTextEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QListWidget {
  background: white;
  border: 1px solid #cfd8d2;
  border-radius: 12px;
  padding: 6px 8px;
}

QListWidget {
  padding: 12px;
}

QScrollArea {
  border: none;
}
"""


def numpy_to_qimage(image: np.ndarray) -> QImage:
    image = core.ensure_uint8_rgb(image)
    if image is None:
        raise ValueError("Image is None.")
    height, width, channels = image.shape
    bytes_per_line = channels * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    rgb = np.array(image.convert("RGB"))
    return QPixmap.fromImage(numpy_to_qimage(rgb))


def load_image_from_path(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


PRIMARY_BRAND_IMAGE = Path("examples/brushnet/src/njut_logo.png")

BRAND_IMAGE_CANDIDATES = (
    Path("examples/brushnet/src/njut_logo.jpg"),
    Path("examples/brushnet/src/njut_logo.jpeg"),
    Path("examples/brushnet/src/nanjing_tech_logo.png"),
    Path("examples/brushnet/src/nanjing_tech_university.png"),
)


def _load_trimmed_brand_pixmap(image_path: Path, target_size: QSize) -> Optional[QPixmap]:
    with Image.open(image_path) as image:
        rgba_image = image.convert("RGBA")
        rgba_array = np.array(rgba_image)

    alpha_mask = rgba_array[:, :, 3] > 0
    content_mask = np.any(rgba_array[:, :, :3] < 245, axis=2)
    non_background_mask = alpha_mask & content_mask
    if not np.any(non_background_mask):
        non_background_mask = alpha_mask
    if not np.any(non_background_mask):
        return None

    rows, cols = np.where(non_background_mask)
    padding = 12
    top = max(0, int(rows.min()) - padding)
    bottom = min(rgba_array.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(cols.min()) - padding)
    right = min(rgba_array.shape[1], int(cols.max()) + padding + 1)

    cropped = Image.fromarray(rgba_array[top:bottom, left:right], mode="RGBA")
    pixmap = pil_to_qpixmap(cropped.convert("RGB"))
    if pixmap.isNull():
        return None
    return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _build_shield_path(rect: QRectF) -> QPainterPath:
    path = QPainterPath()
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    width = rect.width()
    height = rect.height()

    path.moveTo(left + width * 0.12, top)
    path.lineTo(right - width * 0.12, top)
    path.lineTo(right, top + height * 0.11)
    path.lineTo(right, top + height * 0.67)
    path.quadTo(left + width * 0.5, bottom, left, top + height * 0.67)
    path.lineTo(left, top + height * 0.11)
    path.closeSubpath()
    return path


def _draw_brand_pixmap(width: int, height: int) -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)

    brand_blue = QColor("#1b96e3")
    brand_light = QColor("#edf7ff")
    brand_white = QColor("#ffffff")

    shield_width = min(220.0, width * 0.45)
    shield_height = min(240.0, height * 0.58)
    shield_rect = QRectF((width - shield_width) / 2.0, 10.0, shield_width, shield_height)
    inner_rect = shield_rect.adjusted(12.0, 12.0, -12.0, -14.0)

    outer_path = _build_shield_path(shield_rect)
    inner_path = _build_shield_path(inner_rect)

    painter.fillPath(outer_path, brand_white)
    painter.setPen(QPen(brand_blue, 9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawPath(outer_path)

    painter.fillPath(inner_path, brand_white)
    painter.setPen(QPen(brand_blue, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawPath(inner_path)

    top_band = QRectF(inner_rect.left() + 18.0, inner_rect.top() + 10.0, inner_rect.width() - 36.0, inner_rect.height() * 0.30)
    painter.fillRect(top_band, brand_light)
    painter.drawRect(top_band)

    tower_width = top_band.width() * 0.18
    tower_height = top_band.height() * 0.72
    tower_rect = QRectF(
        top_band.center().x() - tower_width / 2.0,
        top_band.top() + 8.0,
        tower_width,
        tower_height,
    )
    painter.setBrush(brand_blue)
    painter.drawRect(tower_rect)
    for offset in (-tower_width * 0.75, 0.0, tower_width * 0.75):
        battlement = QRectF(tower_rect.center().x() + offset - tower_width * 0.18, tower_rect.top() - 6.0, tower_width * 0.36, 8.0)
        painter.drawRect(battlement)

    painter.setPen(QPen(brand_blue, 2))
    painter.setBrush(Qt.NoBrush)

    left_scroll = QPainterPath()
    left_scroll.moveTo(top_band.left() + 18.0, top_band.bottom() - 8.0)
    left_scroll.cubicTo(
        top_band.left() + 6.0,
        top_band.center().y(),
        top_band.left() + 8.0,
        top_band.top() + 16.0,
        top_band.left() + 28.0,
        top_band.top() + 12.0,
    )
    left_scroll.cubicTo(
        top_band.left() + 44.0,
        top_band.top() + 10.0,
        top_band.left() + 40.0,
        top_band.center().y(),
        top_band.left() + 24.0,
        top_band.center().y() + 6.0,
    )
    painter.drawPath(left_scroll)

    right_scroll = QPainterPath()
    right_scroll.moveTo(top_band.right() - 18.0, top_band.bottom() - 8.0)
    right_scroll.cubicTo(
        top_band.right() - 6.0,
        top_band.center().y(),
        top_band.right() - 8.0,
        top_band.top() + 16.0,
        top_band.right() - 28.0,
        top_band.top() + 12.0,
    )
    right_scroll.cubicTo(
        top_band.right() - 44.0,
        top_band.top() + 10.0,
        top_band.right() - 40.0,
        top_band.center().y(),
        top_band.right() - 24.0,
        top_band.center().y() + 6.0,
    )
    painter.drawPath(right_scroll)

    painter.setPen(QPen(brand_blue, 2))
    painter.setFont(QFont("Times New Roman", 12, QFont.Bold))
    painter.drawText(QRectF(tower_rect.left() - 28.0, tower_rect.bottom() + 6.0, tower_rect.width() + 56.0, 22.0), Qt.AlignCenter, "1902")

    book_rect = QRectF(inner_rect.left() + 24.0, inner_rect.top() + inner_rect.height() * 0.42, inner_rect.width() - 48.0, inner_rect.height() * 0.20)
    painter.drawRoundedRect(book_rect, 10.0, 10.0)
    painter.drawLine(QPointF(book_rect.center().x(), book_rect.top() + 4.0), QPointF(book_rect.center().x(), book_rect.bottom() - 4.0))
    painter.drawLine(QPointF(book_rect.left() + 10.0, book_rect.top() + 8.0), QPointF(book_rect.center().x() - 8.0, book_rect.top() + 16.0))
    painter.drawLine(QPointF(book_rect.right() - 10.0, book_rect.top() + 8.0), QPointF(book_rect.center().x() + 8.0, book_rect.top() + 16.0))
    painter.setFont(QFont("STKaiti", 11, QFont.Bold))
    painter.drawText(book_rect.adjusted(6.0, 14.0, -6.0, -4.0), Qt.AlignCenter, "南京工业大学")

    ribbon_rect = QRectF(inner_rect.left() + 34.0, inner_rect.bottom() - 38.0, inner_rect.width() - 68.0, 22.0)
    painter.drawRoundedRect(ribbon_rect, 8.0, 8.0)
    painter.setFont(QFont("Times New Roman", 10, QFont.Bold))
    painter.drawText(ribbon_rect, Qt.AlignCenter, "NANJING TECH")

    chinese_rect = QRectF(12.0, shield_rect.bottom() + 18.0, width - 24.0, 58.0)
    english_rect = QRectF(12.0, chinese_rect.bottom() + 2.0, width - 24.0, 42.0)

    painter.setPen(QPen(brand_blue, 1))
    chinese_font = QFont("STKaiti", 34)
    chinese_font.setBold(True)
    painter.setFont(chinese_font)
    painter.drawText(chinese_rect, Qt.AlignCenter, "南京工业大学")

    english_font = QFont("Times New Roman", 20, QFont.Bold)
    english_font.setLetterSpacing(QFont.AbsoluteSpacing, 1.2)
    painter.setFont(english_font)
    painter.drawText(english_rect, Qt.AlignCenter, "NANJING TECH UNIVERSITY")

    painter.end()
    return pixmap


def build_brand_pixmap(target_size: QSize = QSize(112, 112)) -> QPixmap:
    primary_path = core.PROJECT_ROOT / PRIMARY_BRAND_IMAGE
    if primary_path.exists():
        pixmap = _load_trimmed_brand_pixmap(primary_path, target_size)
        if pixmap is not None:
            return pixmap

    for relative_path in BRAND_IMAGE_CANDIDATES:
        candidate = core.PROJECT_ROOT / relative_path
        if candidate.exists():
            pixmap = _load_trimmed_brand_pixmap(candidate, target_size)
            if pixmap is not None:
                return pixmap
    return _draw_brand_pixmap(target_size.width(), target_size.height())


class ImageCanvas(QLabel):
    imageClicked = pyqtSignal(int, int)

    def __init__(self, title: str, clickable: bool = False, minimum_height: int = 280, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title = title
        self._clickable = clickable
        self._image: Optional[np.ndarray] = None
        self._qimage: Optional[QImage] = None
        self._display_rect = QRect()

        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(minimum_height)
        self.setContentsMargins(10, 10, 10, 10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            """
            QLabel {
              background: #ffffff;
              border: 1px dashed #b8c8bf;
              border-radius: 18px;
              color: #60716b;
              padding: 10px;
            }
            """
        )
        self._set_placeholder()

    def _set_placeholder(self) -> None:
        suffix = "，单击后可在图像上添加分割点" if self._clickable else ""
        self.setText(f"{self._title}\n\n等待载入图像{suffix}")
        self.setPixmap(QPixmap())
        self._display_rect = QRect()

    def set_numpy_image(self, image: Optional[np.ndarray]) -> None:
        self._image = None if image is None else np.array(image, copy=True)
        self._qimage = None if image is None else numpy_to_qimage(image)
        self._update_pixmap()

    def clear_image(self) -> None:
        self._image = None
        self._qimage = None
        self._set_placeholder()

    def _update_pixmap(self) -> None:
        if self._qimage is None:
            self._set_placeholder()
            return

        available = self.contentsRect()
        if available.width() <= 0 or available.height() <= 0:
            return

        pixmap = QPixmap.fromImage(self._qimage)
        scaled = pixmap.scaled(available.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = available.x() + max(0, (available.width() - scaled.width()) // 2)
        y = available.y() + max(0, (available.height() - scaled.height()) // 2)
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setText("")
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        super().mousePressEvent(event)
        if not self._clickable or self._image is None or event.button() != Qt.LeftButton:
            return
        if not self._display_rect.contains(event.pos()):
            return

        display_width = max(1, self._display_rect.width())
        display_height = max(1, self._display_rect.height())
        local_x = event.pos().x() - self._display_rect.x()
        local_y = event.pos().y() - self._display_rect.y()
        image_height, image_width = self._image.shape[:2]
        image_x = int(np.clip(local_x * image_width / display_width, 0, image_width - 1))
        image_y = int(np.clip(local_y * image_height / display_height, 0, image_height - 1))
        self.imageClicked.emit(image_x, image_y)


class InferenceWorker(QThread):
    succeeded = pyqtSignal(object, object, object, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        input_image: Optional[np.ndarray],
        original_image: Optional[np.ndarray],
        original_mask: Optional[np.ndarray],
        uploaded_mask: Optional[np.ndarray],
        selected_points: Sequence[Tuple[Tuple[int, int], int]],
        prompt: str,
        negative_prompt: str,
        blending: bool,
        invert_mask: bool,
        control_strength: float,
        seed: int,
        randomize_seed: bool,
        guidance_scale: float,
        num_inference_steps: int,
        num_outputs: int,
        base_model: str,
        brushnet_path: str,
    ):
        super().__init__()
        self.payload = {
            "input_image": None if input_image is None else np.array(input_image, copy=True),
            "original_image": None if original_image is None else np.array(original_image, copy=True),
            "original_mask": None if original_mask is None else np.array(original_mask, copy=True),
            "uploaded_mask": None if uploaded_mask is None else np.array(uploaded_mask, copy=True),
            "selected_points": list(selected_points),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "blending": blending,
            "invert_mask": invert_mask,
            "control_strength": control_strength,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_outputs": num_outputs,
            "base_model": base_model,
            "brushnet_path": brushnet_path,
        }

    def run(self) -> None:  # type: ignore[override]
        try:
            results, effective_mask, masked_image, status = core.run_brushnet_inference(**self.payload)
        except core.BrushNetAppError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"推理过程中发生异常：{exc}")
        else:
            self.succeeded.emit(results, effective_mask, masked_image, status)


class BrushNetQtWindow(QMainWindow):
    def __init__(self, base_model: str, brushnet_path: str):
        super().__init__()
        self.default_base_model = base_model
        self.default_brushnet_path = brushnet_path

        self.original_image: Optional[np.ndarray] = None
        self.original_mask: Optional[np.ndarray] = None
        self.uploaded_mask: Optional[np.ndarray] = None
        self.selected_points: List[Tuple[Tuple[int, int], int]] = []
        self.current_results: List[Image.Image] = []
        self.worker: Optional[InferenceWorker] = None

        self.setWindowTitle("基于扩散模型的双分支结构的图像修复技术研究 - PyQt5")
        self.resize(1600, 980)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._set_initial_status()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        hero_card = QFrame()
        hero_card.setObjectName("HeroCard")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(18, 12, 18, 12)
        hero_layout.setSpacing(0)

        title_label = QLabel("基于扩散模型的双分支结构的图像修复技术研究")
        title_label.setObjectName("TitleLabel")
        title_label.setWordWrap(True)
        subtitle_label = QLabel("桌面版可视化界面：支持示例加载、SAM 点击分割、黑白掩码上传、模型切换与局部重绘结果展示。")
        subtitle_label.setObjectName("SubTitleLabel")
        subtitle_label.setWordWrap(True)
        subtitle_label.setMaximumWidth(860)

        brand_label = QLabel()
        brand_label.setObjectName("BrandLogoLabel")
        brand_label.setAlignment(Qt.AlignCenter)
        brand_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        brand_pixmap = build_brand_pixmap(QSize(112, 112))
        brand_label.setPixmap(brand_pixmap)
        brand_label.setFixedSize(brand_pixmap.size())

        hero_row = QHBoxLayout()
        hero_row.setSpacing(14)
        hero_row.addWidget(brand_label, 0, Qt.AlignVCenter)

        hero_text_widget = QWidget()
        hero_text_layout = QVBoxLayout(hero_text_widget)
        hero_text_layout.setContentsMargins(0, 0, 0, 0)
        hero_text_layout.setSpacing(2)
        hero_text_layout.addStretch(1)
        hero_text_layout.addWidget(title_label)
        hero_text_layout.addWidget(subtitle_label)
        hero_text_layout.addStretch(1)
        hero_row.addWidget(hero_text_widget, 1)

        hero_layout.addLayout(hero_row)
        root_layout.addWidget(hero_card)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)

        image_group = QGroupBox("输入图像与掩码")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(12)

        self.input_canvas = ImageCanvas("输入图像", clickable=True, minimum_height=520)
        self.input_canvas.imageClicked.connect(self._handle_canvas_click)
        image_layout.addWidget(self.input_canvas)

        image_button_row = QHBoxLayout()
        self.open_image_button = QPushButton("上传图片")
        self.open_mask_button = QPushButton("上传黑白掩码")
        self.clear_mask_button = QPushButton("清空上传掩码")
        self.clear_mask_button.setObjectName("AccentButton")
        image_button_row.addWidget(self.open_image_button)
        image_button_row.addWidget(self.open_mask_button)
        image_button_row.addWidget(self.clear_mask_button)
        image_layout.addLayout(image_button_row)

        example_row = QHBoxLayout()
        self.example_combo = QComboBox()
        self.example_combo.addItems([case["label"] for case in core.EXAMPLE_CASES])
        self.load_example_button = QPushButton("加载示例")
        self.load_example_button.setObjectName("AccentButton")
        example_row.addWidget(self.example_combo, 1)
        example_row.addWidget(self.load_example_button)
        image_layout.addLayout(example_row)

        point_row = QHBoxLayout()
        self.point_mode_combo = QComboBox()
        self.point_mode_combo.addItems(["添加区域", "排除区域"])
        self.invert_mask_checkbox = QCheckBox("反转掩码语义")
        self.invert_mask_checkbox.setChecked(True)
        point_row.addWidget(QLabel("点击模式"))
        point_row.addWidget(self.point_mode_combo, 1)
        point_row.addWidget(self.invert_mask_checkbox)
        image_layout.addLayout(point_row)

        point_button_row = QHBoxLayout()
        self.undo_button = QPushButton("撤销一个点")
        self.clear_points_button = QPushButton("清空点击分割")
        point_button_row.addWidget(self.undo_button)
        point_button_row.addWidget(self.clear_points_button)
        image_layout.addLayout(point_button_row)

        left_layout.addWidget(image_group)

        prompt_group = QGroupBox("提示词与模型参数")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setSpacing(10)

        prompt_layout.addWidget(QLabel("Prompt"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(74)
        self.prompt_edit.setPlaceholderText("例如：A futuristic electric car parked in the city street")
        prompt_layout.addWidget(self.prompt_edit)

        prompt_layout.addWidget(QLabel("Negative Prompt"))
        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setFixedHeight(74)
        self.negative_prompt_edit.setPlainText("ugly, low quality, blurry, distorted")
        prompt_layout.addWidget(self.negative_prompt_edit)

        variant_row = QHBoxLayout()
        self.model_variant_combo = QComboBox()
        self.model_variant_combo.addItems(list(core.get_brushnet_variant_map(self.default_base_model).keys()))
        variant_row.addWidget(QLabel("模型预设"))
        variant_row.addWidget(self.model_variant_combo, 1)
        prompt_layout.addLayout(variant_row)

        prompt_layout.addWidget(QLabel("基础模型路径 / Hugging Face 模型 ID"))
        self.base_model_edit = QLineEdit(self.default_base_model)
        prompt_layout.addWidget(self.base_model_edit)

        prompt_layout.addWidget(QLabel("BrushNet 权重路径"))
        self.brushnet_path_edit = QLineEdit(self.default_brushnet_path)
        prompt_layout.addWidget(self.brushnet_path_edit)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.control_strength_spin = QDoubleSpinBox()
        self.control_strength_spin.setRange(0.0, 1.2)
        self.control_strength_spin.setSingleStep(0.01)
        self.control_strength_spin.setValue(1.0)

        self.guidance_scale_spin = QDoubleSpinBox()
        self.guidance_scale_spin.setRange(1.0, 15.0)
        self.guidance_scale_spin.setSingleStep(0.1)
        self.guidance_scale_spin.setValue(7.5)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 60)
        self.steps_spin.setValue(50)

        self.outputs_spin = QSpinBox()
        self.outputs_spin.setRange(1, 4)
        self.outputs_spin.setValue(1)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(1234)

        self.randomize_seed_checkbox = QCheckBox("每次随机种子")
        self.blending_checkbox = QCheckBox("启用边缘模糊融合")

        grid.addWidget(QLabel("Control Strength"), 0, 0)
        grid.addWidget(self.control_strength_spin, 0, 1)
        grid.addWidget(QLabel("Guidance Scale"), 0, 2)
        grid.addWidget(self.guidance_scale_spin, 0, 3)
        grid.addWidget(QLabel("Inference Steps"), 1, 0)
        grid.addWidget(self.steps_spin, 1, 1)
        grid.addWidget(QLabel("生成结果数"), 1, 2)
        grid.addWidget(self.outputs_spin, 1, 3)
        grid.addWidget(QLabel("Seed"), 2, 0)
        grid.addWidget(self.seed_spin, 2, 1)
        grid.addWidget(self.randomize_seed_checkbox, 2, 2, 1, 2)
        grid.addWidget(self.blending_checkbox, 3, 0, 1, 2)
        prompt_layout.addLayout(grid)

        self.run_button = QPushButton("开始生成")
        self.run_button.setObjectName("PrimaryButton")
        prompt_layout.addWidget(self.run_button)

        left_layout.addWidget(prompt_group)
        left_layout.addStretch(1)

        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)

        preview_group = QGroupBox("掩码预览")
        preview_layout = QHBoxLayout(preview_group)
        preview_layout.setSpacing(12)
        self.mask_preview = ImageCanvas("有效编辑掩码", clickable=False, minimum_height=260)
        self.masked_preview = ImageCanvas("送入模型的 Masked Image", clickable=False, minimum_height=260)
        preview_layout.addWidget(self.mask_preview, 1)
        preview_layout.addWidget(self.masked_preview, 1)
        right_layout.addWidget(preview_group)

        result_group = QGroupBox("生成结果")
        result_layout = QVBoxLayout(result_group)
        self.result_list = QListWidget()
        self.result_list.setViewMode(QListWidget.IconMode)
        self.result_list.setResizeMode(QListWidget.Adjust)
        self.result_list.setWrapping(True)
        self.result_list.setSpacing(12)
        self.result_list.setMovement(QListWidget.Static)
        self.result_list.setIconSize(QSize(260, 260))
        result_layout.addWidget(self.result_list)
        right_layout.addWidget(result_group, 1)

        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        self.status_box = QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(190)
        status_layout.addWidget(self.status_box)
        right_layout.addWidget(status_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([860, 700])

        self.setCentralWidget(central)

        self.open_image_button.clicked.connect(self._choose_source_image)
        self.open_mask_button.clicked.connect(self._choose_mask_image)
        self.clear_mask_button.clicked.connect(self._clear_uploaded_mask)
        self.load_example_button.clicked.connect(self._load_selected_example)
        self.undo_button.clicked.connect(self._undo_last_point)
        self.clear_points_button.clicked.connect(self._clear_all_points)
        self.invert_mask_checkbox.stateChanged.connect(self._refresh_preview)
        self.model_variant_combo.currentTextChanged.connect(self._apply_variant_selection)
        self.run_button.clicked.connect(self._run_inference)

    def _set_initial_status(self) -> None:
        self.status_box.setPlainText(
            core.summarize_status(
                "系统已就绪，等待载入图片。",
                self.default_base_model,
                self.default_brushnet_path,
            )
        )
        self._apply_variant_selection(self.model_variant_combo.currentText())

    def _current_base_model(self) -> str:
        return self.base_model_edit.text().strip() or self.default_base_model

    def _current_brushnet_path(self) -> str:
        return self.brushnet_path_edit.text().strip() or self.default_brushnet_path

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "BrushNet", message)

    def _set_status(self, status: str) -> None:
        self.status_box.setPlainText(status)

    def _set_gallery_results(self, images: Sequence[Image.Image]) -> None:
        self.result_list.clear()
        self.current_results = list(images)
        for index, image in enumerate(images, start=1):
            pixmap = pil_to_qpixmap(image)
            scaled = pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(scaled), f"结果 {index}")
            item.setSizeHint(QSize(280, 300))
            self.result_list.addItem(item)

    def _clear_gallery_results(self) -> None:
        self.current_results = []
        self.result_list.clear()

    def _choose_source_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            str(core.PROJECT_ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return

        try:
            image = load_image_from_path(path)
            payload = core.prepare_source_image(image, self._current_base_model(), self._current_brushnet_path())
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:
            self._show_error(f"读取图片失败：{exc}")
            return

        self.original_image = payload["image"]  # type: ignore[assignment]
        self.original_mask = None
        self.uploaded_mask = None
        self.selected_points = []
        self.input_canvas.set_numpy_image(payload["overlay"])  # type: ignore[arg-type]
        self.mask_preview.clear_image()
        self.masked_preview.set_numpy_image(self.original_image)
        self._clear_gallery_results()
        self._set_status(payload["status"])  # type: ignore[arg-type]
        self.outputs_spin.setValue(1)
        self.steps_spin.setValue(50)
        self.seed_spin.setValue(1234)
        self.control_strength_spin.setValue(1.0)
        self.guidance_scale_spin.setValue(7.5)
        self.randomize_seed_checkbox.setChecked(False)
        self.model_variant_combo.setCurrentText("分割掩码模型")

    def _choose_mask_image(self) -> None:
        if self.original_image is None:
            self._show_error("请先上传原始图片，再上传黑白掩码。")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择黑白掩码图",
            str(core.PROJECT_ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return

        try:
            self.uploaded_mask = load_image_from_path(path)
            self.original_mask = None
            self.selected_points = []
            preview = core.build_preview_from_uploaded_mask(
                self.original_image,
                self.uploaded_mask,
                self.invert_mask_checkbox.isChecked(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:
            self._show_error(f"读取掩码失败：{exc}")
            return

        self.input_canvas.set_numpy_image(preview["overlay"])  # type: ignore[arg-type]
        self.mask_preview.set_numpy_image(preview["effective_mask"])  # type: ignore[arg-type]
        self.masked_preview.set_numpy_image(preview["masked_image"])  # type: ignore[arg-type]
        self._clear_gallery_results()
        self._set_status(preview["status"])  # type: ignore[arg-type]

    def _clear_uploaded_mask(self) -> None:
        if self.uploaded_mask is None:
            return
        self.uploaded_mask = None
        self._refresh_preview()

    def _load_selected_example(self) -> None:
        try:
            payload = core.load_example_case(
                self.example_combo.currentText(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return

        self.original_image = payload["image"]  # type: ignore[assignment]
        self.original_mask = payload["raw_mask"]  # type: ignore[assignment]
        self.uploaded_mask = None
        self.selected_points = []

        self.input_canvas.set_numpy_image(payload["overlay"])  # type: ignore[arg-type]
        self.mask_preview.set_numpy_image(payload["raw_mask"])  # type: ignore[arg-type]
        self.masked_preview.set_numpy_image(payload["masked_image"])  # type: ignore[arg-type]
        self.prompt_edit.setPlainText(payload["prompt"])  # type: ignore[arg-type]
        self.negative_prompt_edit.setPlainText(payload["negative_prompt"])  # type: ignore[arg-type]
        self._set_gallery_results(payload["results"])  # type: ignore[arg-type]
        self.outputs_spin.setValue(1)
        self.steps_spin.setValue(50)
        self.seed_spin.setValue(1234)
        self.control_strength_spin.setValue(1.0)
        self.guidance_scale_spin.setValue(7.5)
        self.randomize_seed_checkbox.setChecked(False)
        self.model_variant_combo.setCurrentText("分割掩码模型")
        self._set_status(payload["status"])  # type: ignore[arg-type]

    def _handle_canvas_click(self, x: int, y: int) -> None:
        if self.original_image is None:
            return

        label = 1 if self.point_mode_combo.currentText() == "添加区域" else 0
        self.selected_points.append(((x, y), label))
        self.uploaded_mask = None

        try:
            preview = core.build_preview_from_points(
                self.original_image,
                self.selected_points,
                self.invert_mask_checkbox.isChecked(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            if self.selected_points:
                self.selected_points.pop()
            return

        self.original_mask = preview["raw_mask"]  # type: ignore[assignment]
        self.input_canvas.set_numpy_image(preview["overlay"])  # type: ignore[arg-type]
        self.mask_preview.set_numpy_image(preview["effective_mask"])  # type: ignore[arg-type]
        self.masked_preview.set_numpy_image(preview["masked_image"])  # type: ignore[arg-type]
        self._set_status(preview["status"])  # type: ignore[arg-type]

    def _undo_last_point(self) -> None:
        if not self.selected_points:
            return
        self.selected_points.pop()
        if not self.selected_points:
            self.original_mask = None
        self._refresh_preview()

    def _clear_all_points(self) -> None:
        if not self.selected_points and self.original_mask is None:
            return
        self.selected_points = []
        self.original_mask = None
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        try:
            preview = core.refresh_mask_preview(
                self.original_image,
                self.original_mask,
                self.uploaded_mask,
                self.selected_points,
                self.invert_mask_checkbox.isChecked(),
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        except core.BrushNetAppError as exc:
            self._show_error(str(exc))
            return

        if preview["overlay"] is None:
            self.input_canvas.clear_image()
        else:
            self.input_canvas.set_numpy_image(preview["overlay"])  # type: ignore[arg-type]

        if preview["effective_mask"] is None:
            self.mask_preview.clear_image()
        else:
            self.mask_preview.set_numpy_image(preview["effective_mask"])  # type: ignore[arg-type]

        if preview["masked_image"] is None:
            self.masked_preview.clear_image()
        else:
            self.masked_preview.set_numpy_image(preview["masked_image"])  # type: ignore[arg-type]

        self._set_status(preview["status"])  # type: ignore[arg-type]

    def _apply_variant_selection(self, variant_name: str) -> None:
        variant_map = core.get_brushnet_variant_map(self._current_base_model())
        selected = variant_map.get(variant_name)
        if not selected:
            return
        self.base_model_edit.setText(selected["base_model"])
        self.brushnet_path_edit.setText(selected["brushnet_path"])

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.run_button.setEnabled(enabled)
        self.open_image_button.setEnabled(enabled)
        self.open_mask_button.setEnabled(enabled)
        self.load_example_button.setEnabled(enabled)

    def _run_inference(self) -> None:
        self._clear_gallery_results()
        self._set_controls_enabled(False)
        self.run_button.setText("生成中...")
        self._set_status(
            core.summarize_status(
                "模型推理已启动，请稍候。",
                self._current_base_model(),
                self._current_brushnet_path(),
                mask_source="处理中",
            )
        )

        self.worker = InferenceWorker(
            input_image=self.original_image,
            original_image=self.original_image,
            original_mask=self.original_mask,
            uploaded_mask=self.uploaded_mask,
            selected_points=self.selected_points,
            prompt=self.prompt_edit.toPlainText(),
            negative_prompt=self.negative_prompt_edit.toPlainText(),
            blending=self.blending_checkbox.isChecked(),
            invert_mask=self.invert_mask_checkbox.isChecked(),
            control_strength=self.control_strength_spin.value(),
            seed=self.seed_spin.value(),
            randomize_seed=self.randomize_seed_checkbox.isChecked(),
            guidance_scale=self.guidance_scale_spin.value(),
            num_inference_steps=self.steps_spin.value(),
            num_outputs=self.outputs_spin.value(),
            base_model=self._current_base_model(),
            brushnet_path=self._current_brushnet_path(),
        )
        self.worker.succeeded.connect(self._handle_inference_success)
        self.worker.failed.connect(self._handle_inference_failure)
        self.worker.finished.connect(self._handle_worker_finished)
        self.worker.start()

    def _handle_inference_success(self, results, effective_mask, masked_image, status: str) -> None:
        if effective_mask is not None and self.original_image is not None:
            overlay_points = self.selected_points if self.uploaded_mask is None else None
            overlay = core.build_overlay_image(self.original_image, effective_mask, overlay_points)
            self.input_canvas.set_numpy_image(overlay)
            self.mask_preview.set_numpy_image(effective_mask)
            self.masked_preview.set_numpy_image(masked_image)
        self._set_gallery_results(results)
        self._set_status(status)

    def _handle_inference_failure(self, message: str) -> None:
        self._show_error(message)
        self._set_status(
            core.summarize_status(
                message,
                self._current_base_model(),
                self._current_brushnet_path(),
            )
        )

    def _handle_worker_finished(self) -> None:
        self._set_controls_enabled(True)
        self.run_button.setText("开始生成")
        self.worker = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BrushNet PyQt5 desktop UI")
    parser.add_argument("--base-model", default=core.resolve_default_base_model(), help="Base model path or model id")
    parser.add_argument("--brushnet-path", default=core.resolve_default_brushnet_path(), help="BrushNet checkpoint path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("BrushNet PyQt5")
    app.setFont(QFont("Microsoft YaHei", 10))

    window = BrushNetQtWindow(args.base_model, args.brushnet_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
