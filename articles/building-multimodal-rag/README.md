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

### Using the pixi env as a kernel in VS Code / another IDE

`pixi run notebook` launches JupyterLab from inside the pixi env, so it already uses the right
kernel. If you instead open the notebook directly in an IDE (e.g. VS Code's Jupyter extension),
register the pixi env as a named ipykernel so it shows up in the kernel picker:

```bash
pixi run python -m ipykernel install --user --name multimodal-rag --display-name "Python (multimodal-rag pixi)"
```

Then in the notebook, select **"Python (multimodal-rag pixi)"** as the kernel (top-right kernel
picker in VS Code, or Kernel → Change Kernel in Jupyter). Re-select it any time the IDE seems to
be running against a stale environment.

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
