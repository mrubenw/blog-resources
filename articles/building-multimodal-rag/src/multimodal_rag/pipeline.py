import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image

from .embedder import CLIPEmbedder, ImageProcessor
from .database import MilvusDatabase
from .rag import RAGProcessor


def load_context_images(image_paths: List[str]) -> List[Image.Image]:
    context_images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            continue
        img = Image.open(image_path)
        context_images.append(img)
    return context_images


def display_images(images: List[Image.Image], figsize: Tuple[int, int] = (15, 5)):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)

    if len(images) == 1:
        axes = [axes]

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f'Image {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def search_and_respond(
    rag_processor: RAGProcessor,
    embedder: CLIPEmbedder,
    database: MilvusDatabase,
    user_text_prompt: str = "",
    user_image_path: str = "",
) -> Optional[Tuple[str, List[Image.Image]]]:
    if not user_text_prompt and not user_image_path:
        raise ValueError("Either user_text_prompt or user_image_path must be provided")

    user_image = None
    if user_image_path:
        try:
            user_image = ImageProcessor.load_image(user_image_path)
        except Exception:
            user_image = Image.open(user_image_path)

    if user_text_prompt and user_image_path:
        query_embedding = embedder.encode_multimodal(user_text_prompt, user_image)
    elif user_text_prompt:
        query_embedding = embedder.encode_text(user_text_prompt).detach().cpu().numpy()
    elif user_image_path:
        query_embedding = embedder.encode_image(user_image).detach().cpu().numpy()

    search_results = database.search_similar(query_embedding.tolist()[0])
    image_paths = [item["entity"]["filename"] for item in search_results]
    context_images = load_context_images(image_paths)

    if not user_text_prompt:
        raise ValueError("Text prompt is required for RAG")

    response = rag_processor.generate_response(user_text_prompt, user_image, context_images)
    print(response)
    display_images(context_images)
    return response, context_images
