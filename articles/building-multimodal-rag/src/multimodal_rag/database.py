from typing import List

from pymilvus import DataType, MilvusClient


class MilvusDatabase:
    EMBEDDING_DIM = 512

    def __init__(self, db_path: str = "./food_tiny.db", collection_name: str = "food_tiny"):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name

    def setup_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        index_params = self.client.prepare_index_params()
        # milvus-lite local mode supports FLAT, HNSW, AUTOINDEX (not IVF_FLAT)
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 64},
        )

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.EMBEDDING_DIM,
        )
        schema.add_field(
            field_name="filename", datatype=DataType.VARCHAR, max_length=500
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_image_data(self, filename: str, embedding: List[float]) -> None:
        self.client.insert(
            collection_name=self.collection_name,
            data={"filename": filename, "embedding": embedding},
        )

    def search_similar(self, query_embedding: List[float], limit: int = 4) -> List[dict]:
        return self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            output_fields=["filename"],
        )[0]

    def collection_has_data(self) -> bool:
        try:
            if not self.client.has_collection(self.collection_name):
                return False
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return int(stats.get("row_count", 0)) > 0
        except Exception:
            return False
