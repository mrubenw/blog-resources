import os

from datasets import load_dataset
from tqdm import tqdm

from .database import MilvusDatabase
from .embedder import CLIPEmbedder


class DatasetProcessor:
    DEFAULT_DATASET = "ruben3010/food101-tiny"

    def __init__(self, embedder: CLIPEmbedder, database: MilvusDatabase, save_dir: str = "./food_images/"):
        self.embedder = embedder
        self.database = database
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def process_food_dataset(self, dataset_name: str = DEFAULT_DATASET) -> None:
        food_dataset = load_dataset(dataset_name)
        image_uuid = 0

        for img in tqdm(food_dataset["train"]["image"]):
            image_uuid += 1
            file_path = f"{self.save_dir}{image_uuid}.jpg"

            image_embedding = self.embedder.encode_image(img)
            img.save(file_path)
            self.database.insert_image_data(file_path, image_embedding.tolist()[0])
