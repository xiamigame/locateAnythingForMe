"""
locateAnythingForMe 核心 API —— 封装 NVlabs/Eagle 模型
"""
import sys
import os
from typing import List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# 将 Eagle 子模块加入 sys.path
_EAGLE_PATH = Path(__file__).resolve().parent.parent / "submodules" / "Eagle" / "Eagle"
if str(_EAGLE_PATH) not in sys.path:
    sys.path.insert(0, str(_EAGLE_PATH))

from eagle import conversation as conversation_lib
from eagle.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from eagle.conversation import conv_templates
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images,
)
from eagle.utils import disable_torch_init

from .config import LocateConfig, get_default_config


@dataclass
class LocateResult:
    """定位 / 描述结果"""
    prompt: str
    output: str
    image_path: Optional[str] = None


class LocateAnything:
    """
    基于 Eagle 的多模态视觉理解 API。

    使用示例:
        la = LocateAnything(model_path="NVEagle/Eagle-X5-13B-Chat")
        result = la.describe("image.jpg", prompt="Describe this image.")
        print(result.output)
    """

    def __init__(self, config: Optional[LocateConfig] = None, **kwargs):
        """
        初始化模型。

        Args:
            config: LocateConfig 配置对象，为 None 时使用默认配置。
            **kwargs: 覆盖 config 中的字段，如 model_path="...", device="cuda" 等。
        """
        if config is None:
            config = get_default_config()

        # 用 kwargs 覆盖配置字段
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        self.config = config

        # 确定 device
        if config.device == "cuda" and not torch.cuda.is_available():
            print("[locateAnything] CUDA 不可用，回退到 CPU")
            self.config.device = "cpu"

        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._context_len = None
        self._model_name = None

    # ── 模型加载 ──────────────────────────────────────────

    def load(self) -> None:
        """显式加载模型（也可在首次调用时自动加载）。"""
        if self._model is not None:
            return

        print(f"[locateAnything] 加载模型: {self.config.model_path}")
        disable_torch_init()

        self._model_name = get_model_name_from_path(self.config.model_path)
        self._tokenizer, self._model, self._image_processor, self._context_len = (
            load_pretrained_model(
                self.config.model_path,
                None,
                self._model_name,
                False,
                False,
            )
        )
        print("[locateAnything] 模型加载完成")

    def _ensure_loaded(self) -> None:
        """确保模型已加载。"""
        if self._model is None:
            self.load()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def image_processor(self):
        self._ensure_loaded()
        return self._image_processor

    @property
    def model_name(self):
        self._ensure_loaded()
        return self._model_name

    # ── 核心方法 ──────────────────────────────────────────

    def generate(
        self,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        prompt: str = "Describe this image.",
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> Union[LocateResult, List[LocateResult]]:
        """
        对一张或多张图像进行理解 / 描述。

        Args:
            image: 图像路径 (str) 或 PIL.Image，或它们的列表。
            prompt: 文本提示。
            temperature: 采样温度，默认使用 config 中的值。
            top_p: nucleus 采样 top_p。
            max_new_tokens: 最大生成 token 数。
            num_beams: beam search 数量。
            do_sample: 是否采样。

        Returns:
            单个 LocateResult 或 LocateResult 列表。
        """
        self._ensure_loaded()

        # 统一为列表
        single_input = not isinstance(image, list)
        images = [image] if single_input else image

        results = []
        for img in images:
            r = self._generate_one(
                img=img,
                prompt=prompt,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                num_beams=num_beams or self.config.num_beams,
                do_sample=do_sample if do_sample is not None else self.config.do_sample,
            )
            results.append(r)

        return results[0] if single_input else results

    def describe(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Describe this image in detail.",
        **kwargs,
    ) -> LocateResult:
        """
        描述图像内容。等同于 generate() 的别名。
        """
        return self.generate(image=image, prompt=prompt, **kwargs)

    def locate(
        self,
        image: Union[str, Image.Image],
        target: str,
        **kwargs,
    ) -> LocateResult:
        """
        在图像中定位指定目标。

        Args:
            image: 图像路径或 PIL.Image。
            target: 要定位的目标描述，如 "a red car", "the person on the left"。

        Returns:
            LocateResult 包含模型输出的定位描述。
        """
        prompt = f"Please locate and describe the position of {target} in this image. Provide bounding box or spatial description if possible."
        return self.generate(image=image, prompt=prompt, **kwargs)

    # ── 内部方法 ──────────────────────────────────────────

    def _generate_one(
        self,
        img: Union[str, Image.Image],
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        num_beams: int,
        do_sample: bool,
    ) -> LocateResult:
        """对单张图像执行推理。"""

        # 加载图像
        if isinstance(img, str):
            pil_image = Image.open(img).convert("RGB")
            image_path = img
        else:
            pil_image = img
            image_path = None

        # 处理 prompt 中的 image token
        if self._model.config.mm_use_im_start_end:
            full_prompt = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt
            )
        else:
            full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        # 构建对话
        conv = conv_templates[self.config.conv_mode].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # 图像 tensor
        image_tensor = process_images([pil_image], self._image_processor, self._model.config)[0]

        # Tokenize
        input_ids = tokenizer_image_token(
            formatted_prompt,
            self._tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )

        device = self.config.device
        input_ids = input_ids.to(device=device, non_blocking=True)
        image_tensor = image_tensor.to(
            dtype=torch.float16 if self.config.torch_dtype == "float16" else torch.float32,
            device=device,
            non_blocking=True,
        )

        # 推理
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids.unsqueeze(0),
                images=image_tensor.unsqueeze(0),
                image_sizes=[pil_image.size],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=self.config.use_cache,
            )

        output_text = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return LocateResult(prompt=prompt, output=output_text, image_path=image_path)
