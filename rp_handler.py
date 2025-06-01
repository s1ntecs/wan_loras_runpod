import os
import base64
import time
import traceback
import uuid
import torch
import tempfile
import requests
import numpy as np

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from huggingface_hub import hf_hub_download

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_download import file
from runpod.serverless.modules.rp_logger import RunPodLogger

from styles import STYLE_URLS, STYLE_NAMES  # ваши словари


# -------------------------------------------------------------
#  Схема входных данных для валидации
# -------------------------------------------------------------



device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_FRAME_RATE = 16

# Будем хранить текущий активный стиль и путь
CURRENT_LORA_NAME = "./loras/wan_SmNmRmC.safetensors"


def calculate_frames(duration, frame_rate):
    raw_frames = round(duration * frame_rate)
    nearest_multiple_of_4 = round(raw_frames / 4) * 4
    return min(nearest_multiple_of_4 + 1, 81)


class Predictor():
    def setup(self):
        """ 
        Загружаем CLIPVisionModel, VAE и сам WanImageToVideoPipeline. 
        Вызывается один раз перед первым predict.
        """
        try:
            self.image_encoder = CLIPVisionModel.from_pretrained(
                model_id, subfolder="image_encoder", torch_dtype=torch.float32
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.float32
            )
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_id,
                vae=self.vae,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16
            ).to(device)
            self.pipe.enable_model_cpu_offload()

            self.vae_scale_factor_spatial  = self.pipe.vae_scale_factor_spatial
            self.vae_scale_factor_temporal = self.pipe.vae_scale_factor_temporal
            
            # Загрузим LoRA по умолчанию, если он есть
            self.pipe.load_lora_weights(CURRENT_LORA_NAME, multiplier=1.0)
            print(f"Model loaded. VAE scales: spatial={self.vae_scale_factor_spatial}, temporal={self.vae_scale_factor_temporal}")
        except Exception as e:
            print("Error loading pipeline:", str(e))
            raise RuntimeError(f"Failed to load pipeline: {str(e)}")

    def _get_local_lora_path(self, lora_style: str) -> str:
        """
        Сформировать локальный путь к файлу LoRA по ключу стиля:
          - ищем в STYLE_NAMES имя файла,
          - проверяем, лежит ли он в ./loras/,
          - если нет — возвращаем None.
        """
        if not lora_style:
            return None

        filename = STYLE_NAMES.get(lora_style)
        if filename is None:
            return None

        # считаем, что локальные лоры лежат в папке ./loras/
        local_path = os.path.join("./loras", filename)
        if os.path.isfile(local_path):
            return local_path
        return None

    def _download_lora_if_needed(self, lora_style: str) -> str:
        """
        Если у нас уже есть локально — вернём путь.
        Если нет и есть ссылка в STYLE_URLS — скачиваем в ./loras/ и вернём путь.
        """
        # 1) Узнаём файл по ключу
        filename = STYLE_NAMES.get(lora_style)
        if filename is None:
            raise RuntimeError(f"Unknown LORA style: {lora_style}")

        target_dir = "./loras"
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, filename)

        # Если файл уже скачан — сразу возвращаем
        if os.path.isfile(local_path):
            return local_path

        # Иначе — скачиваем по URL
        url = STYLE_URLS.get(lora_style)
        if url is None:
            raise RuntimeError(f"No URL found for LORA style: {lora_style}")

        print(f"Downloading LoRA '{lora_style}' from {url} into {local_path} ...")
        # Если ссылка ведёт на HF «blob»-вид, нужно чуть подправить URL, чтобы его можно было прям скачать raw
        if "huggingface.co" in url and "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download LORA from {url}: HTTP {resp.status_code}")

        with open(local_path, "wb") as f:
            f.write(resp.content)

        print(f"Successfully saved LoRA to {local_path}")
        return local_path

    def load_lora(self, lora_style: str, lora_strength: float = 1.0):
        """
        Верхнеуровневая функция «установки» LoRA:
          - Проверяем, меняется ли стиль (global CURRENT_LORA_STYLE).
          - Если нет, то выходим (уже загружен нужный).
          - Если да, то вызываем pipe.unload_lora_weights(), скачиваем/загружаем новую.
        """
        global CURRENT_LORA_NAME

        # Если стиль не передан — ничего не делаем
        if not lora_style:
            return

        # Скачиваем (или берём локальный) нужный файл
        local_path = self._download_lora_if_needed(lora_style)

         # Если стиль не поменялся — нет смысла перезагрузки
        if CURRENT_LORA_NAME == local_path:
            return

        try:
            self.pipe.unload_lora_weights()
        except Exception:
            # возможно, раньше pipe был без LoRA, пропускаем
            pass
        # Устанавливаем через diffusers
        print(f"Loading LoRA weights from local_path = {local_path} (style={lora_style}, strength={lora_strength})")
        self.pipe.load_lora_weights(local_path, multiplier=lora_strength)
        print("LoRA applied.")

        # Обновляем глобальное состояние
        CURRENT_LORA_NAME = local_path

    def predict(
        self,
        image: str,
        prompt: str,
        negative_prompt: str = "low quality, bad quality, blurry, pixelated, watermark",
        lora_style: str = None,
        lora_strength: float = 1.0,
        duration: float = 3.0,
        fps: int = 16,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 28,
        resize_mode: str = "auto",
        seed: int = None
    ) -> str:
        """
        Запускаем генерацию видео и возвращаем Base64.
        """
        # 1) Обновляем LoRA (если передан стиль)
        if lora_style:
            self.load_lora(lora_style, lora_strength)

        # 2) Рассчитываем количество кадров
        num_frames = calculate_frames(duration, MODEL_FRAME_RATE)

        # 3) Собираем генератор
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            seed = np.random.randint(0, 2**30)
            generator = torch.Generator(device=device).manual_seed(seed)

        # 4) Загружаем изображение
        try:
            input_image = load_image(str(image))
        except Exception as e:
            raise RuntimeError(f"Failed to load input image: {str(e)}")

        # 5) Считаем размеры (аналогично ранее)
        mod_value = self.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        if resize_mode == "fixed_square":
            width = height = 512
        else:
            if resize_mode == "auto":
                aspect_ratio = input_image.height / input_image.width
                if 0.9 <= aspect_ratio <= 1.1:
                    width = height = 512
                else:
                    resize_mode = "keep_aspect_ratio"
            if resize_mode == "keep_aspect_ratio":
                max_area = 480 * 832
                aspect_ratio = input_image.height / input_image.width
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width  = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

        # Гарантируем, что делится на 16
        if height % 16 != 0 or width % 16 != 0:
            height = (height // 16) * 16
            width  = (width  // 16) * 16

        input_image = input_image.resize((width, height))

        # 6) Генерируем кадры
        output = self.pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).frames[0]

        # 7) Сохраняем в MP4 и кодируем в Base64
        local_video_path = tempfile.mkdtemp() + "/" + str(uuid.uuid4()) + ".mp4"
        export_to_video(output, str(local_video_path), fps=fps)

        with open(local_video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        os.remove(local_video_path)
        return video_b64


# -------------------------------------------------------------
#  RunPod Handler
# -------------------------------------------------------------
logger = RunPodLogger()
predictor = None


def handler(job):
    global predictor
    try:
        payload = job.get("input", {})
        if predictor is None:
            predictor = Predictor()
            predictor.setup()

        # Скачиваем входное изображение
        image_url = payload["image_url"]
        image_obj = file(image_url)
        image_path = image_obj["file_path"]

        prompt              = payload["prompt"]
        negative_prompt     = payload.get("negative_prompt", "")
        lora_style          = payload.get("lora_style", None)
        lora_strength       = payload.get("lora_strength", 1.0)
        duration            = payload.get("duration", 3.0)
        fps                 = payload.get("fps", 16)
        guidance_scale      = payload.get("guidance_scale", 5.0)
        num_inference_steps = payload.get("num_inference_steps", 28)
        resize_mode         = payload.get("resize_mode", "auto")
        seed                = payload.get("seed", None)

        video_b64 = predictor.predict(
            image=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            lora_style=lora_style,
            lora_strength=lora_strength,
            duration=duration,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            resize_mode=resize_mode,
            seed=seed
        )

        return {"video_base64": video_b64}

    except Exception as e:
        logger.error(f"An exception was raised: {e}")
        return {
            "error": str(e),
            "output": traceback.format_exc(),
            "refresh_worker": True
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})