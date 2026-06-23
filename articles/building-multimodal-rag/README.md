# Building a Multimodal RAG Pipeline

Food image search that understands both text and images — using CLIP for joint embeddings, Milvus Lite for vector search, and Claude for generation.

## Dataset

[ruben3010/food101-tiny](https://huggingface.co/datasets/ruben3010/food101-tiny) on HuggingFace (~2 000 food images, subset of Food-101).

## Setup

```bash
# Install pixi: https://pixi.sh/latest/#installation
pixi install
cp .env.example .env
# fill in ANTHROPIC_API_KEY
```

## Run

```bash
pixi run test       # unit tests (no API key needed)
pixi run notebook   # JupyterLab
```

## Architecture

```
Query (text / image / both)
        │
        ▼
  CLIPEmbedder          # joint text+image embedding space (ViT-B/32)
        │
        ▼
  MilvusDatabase        # cosine similarity search over food image index
        │
        ▼ (top-k images)
  RAGProcessor          # Claude reads retrieved images + query → answer
        │
        ▼
   Response + images
```
