import base64
from io import BytesIO

import clip
import numpy as np
import torch
from PIL import Image


class ImageProcessor:
    @staticmethod
    def load_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class CLIPEmbedder:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(preprocessed)

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)
        return self.model.encode_text(tokens)

    def encode_multimodal(self, text: str, image: Image.Image) -> np.ndarray:
        text_embedding = self.encode_text(text)
        image_embedding = self.encode_image(image)
        return np.mean([
            image_embedding.detach().cpu().numpy(),
            text_embedding.detach().cpu().numpy(),
        ], axis=0)
