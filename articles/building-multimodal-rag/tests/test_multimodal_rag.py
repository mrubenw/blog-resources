import base64
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from multimodal_rag.database import MilvusDatabase
from multimodal_rag.embedder import CLIPEmbedder, ImageProcessor
from multimodal_rag.pipeline import load_context_images, search_and_respond
from multimodal_rag.rag import RAGProcessor


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------


class TestImageProcessor:
    def test_image_to_base64_roundtrip(self, rgb_image):
        b64 = ImageProcessor.image_to_base64(rgb_image)
        decoded = base64.b64decode(b64)
        restored = Image.open(BytesIO(decoded))
        assert restored.mode == "RGB"

    def test_image_to_base64_converts_rgba(self):
        rgba = Image.new("RGBA", (32, 32), color=(100, 150, 200, 255))
        b64 = ImageProcessor.image_to_base64(rgba)
        decoded = base64.b64decode(b64)
        restored = Image.open(BytesIO(decoded))
        assert restored.mode == "RGB"

    def test_load_image_rgb(self, tmp_path, rgb_image):
        path = str(tmp_path / "test.jpg")
        rgb_image.save(path)
        loaded = ImageProcessor.load_image(path)
        assert loaded.mode == "RGB"
        assert loaded.size == rgb_image.size


# ---------------------------------------------------------------------------
# CLIPEmbedder
# ---------------------------------------------------------------------------


class TestCLIPEmbedder:
    @patch("multimodal_rag.embedder.clip")
    @patch("multimodal_rag.embedder.torch")
    def test_init_loads_model(self, mock_torch, mock_clip):
        mock_torch.cuda.is_available.return_value = False
        mock_clip.load.return_value = (MagicMock(), MagicMock())
        CLIPEmbedder(model_name="ViT-B/32")
        mock_clip.load.assert_called_once_with("ViT-B/32", device="cpu")

    @patch("multimodal_rag.embedder.clip")
    @patch("multimodal_rag.embedder.torch")
    def test_encode_image_calls_model(self, mock_torch, mock_clip, rgb_image):
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_preprocess.return_value.unsqueeze.return_value.to.return_value = MagicMock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        embedder = CLIPEmbedder()
        embedder.encode_image(rgb_image)

        assert mock_model.encode_image.called

    @patch("multimodal_rag.embedder.clip")
    @patch("multimodal_rag.embedder.torch")
    def test_encode_text_calls_model(self, mock_torch, mock_clip):
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_clip.load.return_value = (mock_model, MagicMock())
        mock_clip.tokenize.return_value.to.return_value = MagicMock()

        embedder = CLIPEmbedder()
        embedder.encode_text("a bowl of ramen")

        assert mock_model.encode_text.called


# ---------------------------------------------------------------------------
# MilvusDatabase
# ---------------------------------------------------------------------------


class TestMilvusDatabase:
    def test_setup_collection_empty(self, tmp_path):
        db = MilvusDatabase(db_path=str(tmp_path / "test.db"), collection_name="test_col")
        db.setup_collection()
        assert db.collection_has_data() is False

    def test_insert_and_search(self, tmp_path, flat_embedding):
        db = MilvusDatabase(db_path=str(tmp_path / "test.db"), collection_name="test_col")
        db.setup_collection()
        db.insert_image_data("food/001.jpg", flat_embedding)

        results = db.search_similar(flat_embedding, limit=1)
        assert len(results) == 1
        assert results[0]["entity"]["filename"] == "food/001.jpg"

    def test_collection_has_data_after_insert(self, tmp_path, flat_embedding):
        db = MilvusDatabase(db_path=str(tmp_path / "test.db"), collection_name="test_col")
        db.setup_collection()
        db.insert_image_data("food/001.jpg", flat_embedding)
        assert db.collection_has_data() is True

    def test_collection_has_data_no_collection(self, tmp_path):
        db = MilvusDatabase(db_path=str(tmp_path / "test.db"), collection_name="nonexistent")
        assert db.collection_has_data() is False

    def test_search_returns_closest_match(self, tmp_path):
        db = MilvusDatabase(db_path=str(tmp_path / "test.db"), collection_name="test_col")
        db.setup_collection()

        emb_a = [1.0] + [0.0] * 511
        emb_b = [0.0] + [1.0] + [0.0] * 510
        db.insert_image_data("a.jpg", emb_a)
        db.insert_image_data("b.jpg", emb_b)

        results = db.search_similar(emb_a, limit=1)
        assert results[0]["entity"]["filename"] == "a.jpg"


# ---------------------------------------------------------------------------
# RAGProcessor
# ---------------------------------------------------------------------------


class TestRAGProcessor:
    def test_generate_response_text_only(self, rgb_image):
        with patch("multimodal_rag.rag.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text="Steak goes well with fries and a green salad.")
            ]

            processor = RAGProcessor()
            result = processor.generate_response(
                user_prompt="What goes with steak?",
                user_image=None,
                context_images=[rgb_image, rgb_image],
            )

            assert result == "Steak goes well with fries and a green salad."
            mock_client.messages.create.assert_called_once()

    def test_generate_response_with_user_image(self, rgb_image):
        with patch("multimodal_rag.rag.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [
                MagicMock(text="This looks like a bruschetta.")
            ]

            processor = RAGProcessor()
            processor.generate_response(
                user_prompt="What is this?",
                user_image=rgb_image,
                context_images=[rgb_image],
            )

            call_args = mock_client.messages.create.call_args
            content = call_args.kwargs["messages"][0]["content"]
            image_blocks = [b for b in content if b.get("type") == "image"]
            assert len(image_blocks) == 2  # 1 user image + 1 context image

    def test_uses_claude_sonnet(self):
        with patch("multimodal_rag.rag.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create.return_value.content = [MagicMock(text="ok")]

            processor = RAGProcessor()
            processor.generate_response("query", None, [Image.new("RGB", (32, 32))])

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Pipeline (standalone functions)
# ---------------------------------------------------------------------------


class TestLoadContextImages:
    def test_loads_existing_images(self, tmp_path, rgb_image):
        path = str(tmp_path / "img.jpg")
        rgb_image.save(path)
        images = load_context_images([path])
        assert len(images) == 1

    def test_skips_missing_images(self, tmp_path, rgb_image, capsys):
        real = str(tmp_path / "real.jpg")
        rgb_image.save(real)
        images = load_context_images([real, "/nonexistent/missing.jpg"])
        assert len(images) == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestSearchAndRespond:
    def test_raises_with_no_input(self):
        with pytest.raises(ValueError, match="user_text_prompt or user_image_path"):
            search_and_respond(MagicMock(), MagicMock(), MagicMock())

    def test_raises_without_text_for_rag(self, tmp_path, rgb_image):
        img_path = str(tmp_path / "q.jpg")
        rgb_image.save(img_path)

        mock_emb = MagicMock()
        mock_emb.detach.return_value.cpu.return_value.numpy.return_value = np.zeros((1, 512))
        mock_embedder = MagicMock()
        mock_embedder.encode_image.return_value = mock_emb

        mock_db = MagicMock()
        mock_db.search_similar.return_value = []

        with pytest.raises(ValueError, match="Text prompt is required"):
            search_and_respond(MagicMock(), mock_embedder, mock_db, user_image_path=img_path)

    def test_returns_response_and_images(self, tmp_path, rgb_image, flat_embedding):
        ctx_path = str(tmp_path / "ctx.jpg")
        rgb_image.save(ctx_path)

        mock_emb_tensor = MagicMock()
        mock_emb_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [flat_embedding]
        )
        mock_embedder = MagicMock()
        mock_embedder.encode_text.return_value = mock_emb_tensor

        mock_db = MagicMock()
        mock_db.search_similar.return_value = [{"entity": {"filename": ctx_path}}]

        mock_rag = MagicMock()
        mock_rag.generate_response.return_value = "Steak goes with fries."

        with patch("multimodal_rag.pipeline.display_images"):
            response, images = search_and_respond(
                mock_rag, mock_embedder, mock_db,
                user_text_prompt="What goes with steak?",
            )

        assert response == "Steak goes with fries."
        assert len(images) == 1
