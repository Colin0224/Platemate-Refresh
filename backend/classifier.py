from PIL import Image
from transformers import pipeline
from functools import lru_cache

from .config import get_settings


@lru_cache(maxsize=1)
def _load_pipeline():
    settings = get_settings()
    return pipeline("image-classification", model=settings.classification_model)


def classify_image(image: Image.Image, top_k: int = 5) -> list[dict]:
    pipe = _load_pipeline()
    predictions = pipe(image)
    return [
        {"label": p["label"], "confidence": round(p["score"], 4)}
        for p in predictions[:top_k]
    ]
