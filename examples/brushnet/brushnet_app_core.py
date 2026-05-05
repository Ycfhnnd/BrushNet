from __future__ import annotations

import gc
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler
from segment_anything import SamPredictor, sam_model_registry

os.chdir(PROJECT_ROOT)

# 统一在核心层确定运行设备和默认模型位置，界面层只负责收集参数和展示结果。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAM_CHECKPOINT = PROJECT_ROOT / "data" / "ckpt" / "sam_vit_h_4b8939.pth"
DEFAULT_REMOTE_BASE_MODEL = "runwayml/stable-diffusion-v1-5"
DISABLE_SAFETY_CHECKER = True

POINT_COLORS = [(28, 163, 131), (215, 82, 54)]
POINT_MARKERS = [1, 5]

SEGMENTATION_BRUSHNET_DIR = PROJECT_ROOT / "data" / "ckpt" / "segmentation_mask_brushnet_ckpt"
RANDOM_MASK_BRUSHNET_DIR = PROJECT_ROOT / "data" / "ckpt" / "random_mask_brushnet_ckpt"

# 管线与 SAM 模型都比较重，这里通过缓存避免重复初始化。
PIPELINE_CACHE: Dict[str, Optional[object]] = {
    "pipe": None,
    "base_model": None,
    "brushnet_path": None,
}
PIPELINE_LOCK = threading.Lock()
SAM_STATE: Dict[str, Optional[object]] = {"model": None, "predictor": None}

PointRecord = Tuple[Tuple[int, int], int]


class BrushNetAppError(RuntimeError):
    pass


def normalize_model_reference(value: str) -> str:
    # 同时兼容 Hugging Face 模型 ID、本地相对路径和本地绝对路径。
    stripped = (value or "").strip()
    if not stripped:
        return stripped

    candidate = Path(stripped)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if candidate.exists():
        return str(candidate)
    return stripped


def resolve_default_brushnet_path() -> str:
    preferred = PROJECT_ROOT / "data" / "ckpt" / "segmentation_mask_brushnet_ckpt"
    fallback = PROJECT_ROOT / "data" / "ckpt"
    fallback_weight_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    )

    if preferred.exists():
        return str(preferred)

    if (fallback / "config.json").exists() and any((fallback / name).exists() for name in fallback_weight_names):
        return str(fallback)

    return str(preferred)


def resolve_default_base_model() -> str:
    env_value = os.environ.get("BRUSHNET_BASE_MODEL")
    if env_value:
        return env_value

    ckpt_dir = PROJECT_ROOT / "data" / "ckpt"
    candidate_names = [
        "realisticVisionV60B1_v51VAE",
        "stable-diffusion-v1-5",
        "runwayml--stable-diffusion-v1-5",
    ]

    for name in candidate_names:
        candidate = ckpt_dir / name
        if (candidate / "model_index.json").exists():
            return str(candidate)

    for candidate in ckpt_dir.glob("*"):
        if candidate.is_dir() and (candidate / "model_index.json").exists():
            return str(candidate)

    return DEFAULT_REMOTE_BASE_MODEL


def get_brushnet_variant_map(base_model: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    # 给界面下拉框提供“方案名 -> 实际模型路径”的映射关系。
    resolved_base_model = normalize_model_reference(base_model or "") or resolve_default_base_model()
    default_brushnet = normalize_model_reference(resolve_default_brushnet_path())
    segmentation_path = str(SEGMENTATION_BRUSHNET_DIR) if SEGMENTATION_BRUSHNET_DIR.exists() else default_brushnet
    random_mask_path = str(RANDOM_MASK_BRUSHNET_DIR) if RANDOM_MASK_BRUSHNET_DIR.exists() else default_brushnet

    return {
        "分割掩膜 BrushNet": {
            "base_model": resolved_base_model,
            "brushnet_path": segmentation_path,
        },
        "随机掩膜 BrushNet": {
            "base_model": resolved_base_model,
            "brushnet_path": random_mask_path,
        },
    }


def summarize_status(
    message: str,
    base_model: str,
    brushnet_path: str,
    mask_source: str = "未选择",
    actual_seed: Optional[int] = None,
    elapsed: Optional[float] = None,
) -> str:
    resolved_base = normalize_model_reference(base_model) or resolve_default_base_model()
    resolved_brushnet = normalize_model_reference(brushnet_path) or resolve_default_brushnet_path()

    lines = [
        "BrushNet 运行状态",
        f"运行设备: {DEVICE}",
        f"基础模型: {resolved_base}",
        f"BrushNet 权重: {resolved_brushnet}",
        f"SAM 权重: {'已找到' if SAM_CHECKPOINT.exists() else '未找到'}",
        f"已关闭安全检查器: {DISABLE_SAFETY_CHECKER}",
        f"掩膜来源: {mask_source}",
    ]
    if actual_seed is not None:
        lines.append(f"随机种子: {actual_seed}")
    if elapsed is not None:
        lines.append(f"推理耗时: {elapsed:.2f}s")
    lines.append("")
    lines.append(f"状态说明: {message}")
    return "\n".join(lines)


def ensure_uint8_rgb(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    # 将输入统一整理为三通道 uint8 RGB，便于 OpenCV / PIL / 模型共用。
    if image is None:
        return None
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resize_for_workflow(image: np.ndarray, max_short_side: int = 768) -> np.ndarray:
    # 控制输入尺寸和分辨率对齐，既节省显存也减少后续尺寸不兼容问题。
    image = ensure_uint8_rgb(image)
    assert image is not None

    height, width = image.shape[:2]
    short_side = min(height, width)
    if short_side <= max_short_side:
        return image

    scale = max_short_side / float(short_side)
    new_height = int(np.round(height * scale / 64.0)) * 64
    new_width = int(np.round(width * scale / 64.0)) * 64
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def validate_source_image(image: np.ndarray) -> np.ndarray:
    image = resize_for_workflow(image)
    height, width = image.shape[:2]
    ratio = max(height, width) / max(1, min(height, width))
    if ratio > 2.0:
        raise BrushNetAppError("Image aspect ratio cannot be larger than 2:1.")
    return image


def build_binary_mask(mask_image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    # 将任意掩膜图规整成与原图同尺寸的三通道二值掩膜。
    mask_image = ensure_uint8_rgb(mask_image)
    assert mask_image is not None

    target_width, target_height = target_size
    if mask_image.shape[0] != target_height or mask_image.shape[1] != target_width:
        mask_image = cv2.resize(mask_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    binary = np.where(mask_image.sum(axis=-1) > 255, 255, 0).astype(np.uint8)
    return np.repeat(binary[:, :, None], 3, axis=2)


def maybe_invert_mask(mask_image: np.ndarray, invert_mask: bool) -> np.ndarray:
    return 255 - mask_image if invert_mask else mask_image


def build_overlay_image(
    image: np.ndarray,
    edit_mask: np.ndarray,
    points: Optional[Sequence[PointRecord]] = None,
) -> np.ndarray:
    # 仅用于界面预览：把编辑区域高亮，并绘制点击点。
    image = ensure_uint8_rgb(image)
    assert image is not None

    overlay = image.astype(np.float32).copy()
    binary = edit_mask[:, :, 0] > 127
    tint = np.array([245, 134, 52], dtype=np.float32)
    overlay[binary] = overlay[binary] * 0.35 + tint * 0.65

    if points:
        for point, label in points:
            draw_point = tuple(int(v) for v in point)
            cv2.drawMarker(
                overlay,
                draw_point,
                POINT_COLORS[label],
                markerType=POINT_MARKERS[label],
                markerSize=18,
                thickness=4,
            )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_masked_image(image: np.ndarray, edit_mask: np.ndarray) -> np.ndarray:
    # 构造送给 BrushNet 的输入图：保留未编辑区域，编辑区域置黑。
    image = ensure_uint8_rgb(image)
    assert image is not None

    binary = (edit_mask[:, :, 0] > 127).astype(np.float32)[:, :, None]
    return np.clip(image * (1.0 - binary), 0, 255).astype(np.uint8)


def clear_cached_pipeline() -> None:
    # 切换模型或释放资源时，主动清理旧管线和 GPU 显存。
    pipe = PIPELINE_CACHE.get("pipe")
    if pipe is not None:
        PIPELINE_CACHE["pipe"] = None
        PIPELINE_CACHE["base_model"] = None
        PIPELINE_CACHE["brushnet_path"] = None
        del pipe
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def _brushnet_checkpoint_looks_valid(path_value: str) -> bool:
    candidate = Path(path_value)
    if not candidate.exists() or not candidate.is_dir():
        return False

    weight_names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    )
    return (candidate / "config.json").exists() and any((candidate / name).exists() for name in weight_names)


def _base_model_looks_valid(path_value: str) -> bool:
    candidate = Path(path_value)
    return candidate.exists() and candidate.is_dir() and (candidate / "model_index.json").exists()


def ensure_pipeline(base_model: str, brushnet_path: str) -> Tuple[StableDiffusionBrushNetPipeline, bool, str, str]:
    # 统一负责 BrushNet 管线的合法性检查、懒加载和缓存复用。
    resolved_base = normalize_model_reference(base_model) or resolve_default_base_model()
    resolved_brushnet = normalize_model_reference(brushnet_path) or resolve_default_brushnet_path()

    with PIPELINE_LOCK:
        # 相同配置直接复用缓存，避免重复加载大型模型。
        cached_pipe = PIPELINE_CACHE.get("pipe")
        if (
            cached_pipe is not None
            and PIPELINE_CACHE.get("base_model") == resolved_base
            and PIPELINE_CACHE.get("brushnet_path") == resolved_brushnet
        ):
            return cached_pipe, False, resolved_base, resolved_brushnet

        clear_cached_pipeline()
        if not _brushnet_checkpoint_looks_valid(resolved_brushnet):
            raise BrushNetAppError(
                "BrushNet 权重目录无效。\n"
                f"路径: {resolved_brushnet}\n"
                "目录中应包含 config.json 和模型权重文件。"
            )

        base_candidate = Path(resolved_base)
        if base_candidate.exists() and not _base_model_looks_valid(resolved_base):
            raise BrushNetAppError(
                "基础模型目录无效。\n"
                f"路径: {resolved_base}\n"
                "目录中应包含 diffusers 格式的 model_index.json。"
            )

        try:
            brushnet = BrushNetModel.from_pretrained(resolved_brushnet, torch_dtype=WEIGHT_DTYPE)
            pipe_kwargs = {
                "brushnet": brushnet,
                "torch_dtype": WEIGHT_DTYPE,
                "low_cpu_mem_usage": False,
            }
            if DISABLE_SAFETY_CHECKER:
                pipe_kwargs["safety_checker"] = None
                pipe_kwargs["requires_safety_checker"] = False
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                resolved_base,
                **pipe_kwargs,
            )
        except Exception as exc:
            raise BrushNetAppError(
                "BrushNet 管线初始化失败。\n"
                f"基础模型: {resolved_base}\n"
                f"BrushNet 权重: {resolved_brushnet}\n"
                f"错误信息: {exc}"
            ) from exc

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if DEVICE == "cuda":
            # 在桌面端环境中，CPU offload 更适合控制显存占用。
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(DEVICE)

        PIPELINE_CACHE["pipe"] = pipe
        PIPELINE_CACHE["base_model"] = resolved_base
        PIPELINE_CACHE["brushnet_path"] = resolved_brushnet
        return pipe, True, resolved_base, resolved_brushnet


def ensure_sam_predictor() -> SamPredictor:
    # SAM 只在首次需要交互分割时初始化一次。
    predictor = SAM_STATE.get("predictor")
    if predictor is not None:
        return predictor

    if not SAM_CHECKPOINT.exists():
        raise BrushNetAppError(
            "未找到 SAM 权重文件，期望路径为 data/ckpt/sam_vit_h_4b8939.pth。"
        )

    try:
        sam_model = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT)).to(DEVICE)
        sam_model.eval()
        predictor = SamPredictor(sam_model)
    except Exception as exc:
        raise BrushNetAppError(
            f"SAM 分割器初始化失败: {exc}"
        ) from exc

    SAM_STATE["model"] = sam_model
    SAM_STATE["predictor"] = predictor
    return predictor


def infer_mask_from_points(image: np.ndarray, selected_points: Sequence[PointRecord]) -> np.ndarray:
    # 将前景/背景点击点送入 SAM，得到当前分割掩膜。
    if not selected_points:
        raise BrushNetAppError("请至少提供一个分割点。")

    predictor = ensure_sam_predictor()
    point_coords = np.array([point for point, _ in selected_points], dtype=np.float32)
    point_labels = np.array([label for _, label in selected_points], dtype=np.int32)
    predictor.set_image(image)
    with torch.no_grad():
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    binary = np.where(masks[0], 255, 0).astype(np.uint8)
    return np.repeat(binary[:, :, None], 3, axis=2)


def derive_effective_mask(
    original_image: np.ndarray,
    original_mask: Optional[np.ndarray],
    uploaded_mask: Optional[np.ndarray],
    selected_points: Sequence[PointRecord],
    invert_mask: bool,
) -> Tuple[np.ndarray, str]:
    # 掩膜优先级：
    # 1. 用户上传的掩膜
    # 2. 已保存的原始掩膜
    # 3. 基于当前点击点实时生成的 SAM 掩膜
    if uploaded_mask is not None:
        raw_mask = build_binary_mask(uploaded_mask, (original_image.shape[1], original_image.shape[0]))
        return maybe_invert_mask(raw_mask, invert_mask), "上传掩膜"

    if original_mask is not None:
        raw_mask = build_binary_mask(original_mask, (original_image.shape[1], original_image.shape[0]))
        mask_source = "SAM 掩膜" if selected_points else "已有掩膜"
        return maybe_invert_mask(raw_mask, invert_mask), mask_source

    if selected_points:
        raw_mask = infer_mask_from_points(original_image, selected_points)
        return maybe_invert_mask(raw_mask, invert_mask), "SAM 掩膜"

    raise BrushNetAppError("请先上传掩膜图，或者在图像上点击分割点。")


def prepare_source_image(image: np.ndarray, base_model: str, brushnet_path: str) -> Dict[str, object]:
    # 原图载入后的统一入口：做尺寸合法化，并初始化预览状态。
    prepared_image = validate_source_image(image)
    status = summarize_status(
        "原图已加载，可以继续上传掩膜，或者点击图像进行 SAM 分割。",
        base_model,
        brushnet_path,
        mask_source="未选择",
    )
    return {
        "image": prepared_image,
        "overlay": prepared_image,
        "status": status,
    }


def build_preview_from_uploaded_mask(
    original_image: np.ndarray,
    uploaded_mask: np.ndarray,
    invert_mask: bool,
    base_model: str,
    brushnet_path: str,
) -> Dict[str, object]:
    # 用户直接上传掩膜时，生成界面预览所需的图像和状态信息。
    if original_image is None:
        raise BrushNetAppError("请先加载原图，再上传掩膜。")

    original_image = ensure_uint8_rgb(original_image)
    assert original_image is not None

    effective_mask = maybe_invert_mask(
        build_binary_mask(uploaded_mask, (original_image.shape[1], original_image.shape[0])),
        invert_mask,
    )
    overlay = build_overlay_image(original_image, effective_mask)
    masked_image = build_masked_image(original_image, effective_mask)
    status = summarize_status(
        "已将上传掩膜应用到当前图像。",
        base_model,
        brushnet_path,
        mask_source="上传掩膜",
    )
    return {
        "overlay": overlay,
        "effective_mask": effective_mask,
        "masked_image": masked_image,
        "status": status,
    }


def build_preview_from_points(
    original_image: np.ndarray,
    selected_points: Sequence[PointRecord],
    invert_mask: bool,
    base_model: str,
    brushnet_path: str,
) -> Dict[str, object]:
    # 用户通过点击点交互分割时，生成界面预览所需的图像和状态信息。
    if original_image is None:
        raise BrushNetAppError("请先加载原图，再选择分割点。")
    if not selected_points:
        raise BrushNetAppError("请至少选择一个前景点或背景点。")

    original_image = ensure_uint8_rgb(original_image)
    assert original_image is not None

    raw_mask = infer_mask_from_points(original_image, selected_points)
    effective_mask = maybe_invert_mask(raw_mask, invert_mask)
    overlay = build_overlay_image(original_image, effective_mask, selected_points)
    masked_image = build_masked_image(original_image, effective_mask)
    status = summarize_status(
        "已根据当前选点更新 SAM 分割预览。",
        base_model,
        brushnet_path,
        mask_source="SAM 掩膜",
    )
    return {
        "overlay": overlay,
        "raw_mask": raw_mask,
        "effective_mask": effective_mask,
        "masked_image": masked_image,
        "status": status,
    }


def refresh_mask_preview(
    original_image: Optional[np.ndarray],
    original_mask: Optional[np.ndarray],
    uploaded_mask: Optional[np.ndarray],
    selected_points: Sequence[PointRecord],
    invert_mask: bool,
    base_model: str,
    brushnet_path: str,
) -> Dict[str, object]:
    # 当用户撤销点、切换反转、清空掩膜时，统一从当前状态重新构造预览。
    if original_image is None:
        status = summarize_status("请先加载原图。", base_model, brushnet_path)
        return {
            "overlay": None,
            "effective_mask": None,
            "masked_image": None,
            "status": status,
        }

    image = ensure_uint8_rgb(original_image)
    assert image is not None

    try:
        effective_mask, mask_source = derive_effective_mask(
            image,
            original_mask,
            uploaded_mask,
            selected_points,
            invert_mask,
        )
    except BrushNetAppError:
        status = summarize_status(
            "当前还没有可用掩膜，请上传掩膜或选择分割点。",
            base_model,
            brushnet_path,
        )
        return {
            "overlay": image,
            "effective_mask": None,
            "masked_image": image,
            "status": status,
        }

    overlay_points = selected_points if mask_source == "SAM 掩膜" and uploaded_mask is None else None
    overlay = build_overlay_image(image, effective_mask, overlay_points)
    masked_image = build_masked_image(image, effective_mask)
    status = summarize_status(
        "掩膜预览已刷新。",
        base_model,
        brushnet_path,
        mask_source=mask_source,
    )
    return {
        "overlay": overlay,
        "effective_mask": effective_mask,
        "masked_image": masked_image,
        "status": status,
    }


def run_brushnet_inference(
    input_image: Optional[np.ndarray],
    original_image: Optional[np.ndarray],
    original_mask: Optional[np.ndarray],
    uploaded_mask: Optional[np.ndarray],
    selected_points: Sequence[PointRecord],
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
) -> Tuple[List[Image.Image], np.ndarray, np.ndarray, str]:
    # 核心推理入口：
    # 原图/掩膜准备 -> 获取或初始化管线 -> Batch 推理 -> 可选边缘融合 -> 返回结果。
    source_image = ensure_uint8_rgb(original_image if original_image is not None else input_image)
    if source_image is None:
        raise BrushNetAppError("请先加载原图。")
    if not (prompt or "").strip():
        raise BrushNetAppError("请先输入提示词，再执行推理。")

    effective_mask, mask_source = derive_effective_mask(
        source_image,
        original_mask,
        uploaded_mask,
        selected_points,
        invert_mask,
    )
    masked_image = build_masked_image(source_image, effective_mask)

    if blending and control_strength < 1.0:
        raise BrushNetAppError("启用模糊融合时，Control Strength 必须大于等于 1.0。")

    pipe, _, resolved_base, resolved_brushnet = ensure_pipeline(base_model, brushnet_path)
    actual_seed = random.randint(0, 2_147_483_647) if randomize_seed else int(seed)
    generator = torch.Generator(DEVICE).manual_seed(actual_seed)

    init_image = Image.fromarray(masked_image).convert("RGB")
    mask_image = Image.fromarray(effective_mask).convert("RGB")

    # 支持一次生成多张图，因此需要把 prompt 扩成 batch。
    prompt_batch = [prompt.strip()] * int(num_outputs)
    negative_text = (negative_prompt or "").strip()
    negative_batch = [negative_text] * int(num_outputs) if negative_text else None

    start = time.perf_counter()
    results = pipe(
        prompt_batch,
        init_image,
        mask_image,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        brushnet_conditioning_scale=float(control_strength),
        negative_prompt=negative_batch,
    ).images
    elapsed = time.perf_counter() - start

    if blending:
        blended_results = []
        mask_np = (effective_mask[:, :, 0] > 127).astype(np.float32)[:, :, None]
        # 对掩膜边缘做模糊，减轻生成结果与原图拼接时的接缝感。
        mask_blurred = cv2.GaussianBlur(mask_np[:, :, 0] * 255, (21, 21), 0) / 255.0
        soft_mask = 1 - (1 - mask_np) * (1 - mask_blurred[:, :, None])
        for image in results:
            image_np = np.array(image)
            pasted = source_image * (1 - soft_mask) + image_np * soft_mask
            blended_results.append(Image.fromarray(np.clip(pasted, 0, 255).astype(np.uint8)))
        results = blended_results

    status = summarize_status(
        "推理已完成。",
        resolved_base,
        resolved_brushnet,
        mask_source=mask_source,
        actual_seed=actual_seed,
        elapsed=elapsed,
    )
    return results, effective_mask, masked_image, status
