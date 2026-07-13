# Three RAG Evaluation Metrics

LLM-as-a-Judge evaluation of a multimodal RAG pipeline — measuring faithfulness, context relevance, and answer relevance with Claude Sonnet 4.6.

This project extends [`../building-multimodal-rag`](../building-multimodal-rag): rather than
hand-picking test images or writing down a fake answer, it imports that project's
`multimodal_rag` package (`CLIPEmbedder`, `MilvusDatabase`, `RAGProcessor`) and reuses its
already-built vector index, so every answer and every retrieved image fed to the judge is a
genuine output of the real pipeline. See [How the test scenarios are built](#how-the-test-scenarios-are-built)
below for the exact process.

## Dataset

[ruben3010/food101-tiny](https://huggingface.co/datasets/ruben3010/food101-tiny) on HuggingFace (~2 000 food images, subset of Food-101) — ingested once by `building-multimodal-rag`, then reused here.

## Setup

1. **Build the index in `building-multimodal-rag` first** (skip if you've already run it):

   ```bash
   cd ../building-multimodal-rag
   pixi install
   cp .env.example .env   # fill in ANTHROPIC_API_KEY
   pixi run notebook
   # run multimodal_rag.ipynb top to bottom — it ingests food101-tiny and creates
   # notebooks/food_tiny.db + notebooks/food_images/
   ```

2. **Install this project.** Its `pyproject.toml` declares `multimodal-rag` as an editable path
   dependency on `../building-multimodal-rag`, so `pixi install` here pulls in that package
   (and its CLIP/Milvus/torch dependencies) automatically:

   ```bash
   cd ../three-rag-evaluation-metrics
   # Install pixi: https://pixi.sh/latest/#installation
   pixi install
   cp .env.example .env
   # fill in ANTHROPIC_API_KEY
   ```

## Run

```bash
pixi run notebook   # JupyterLab
```

The first code cell in `notebooks/rag_evaluation.ipynb` points `MilvusDatabase` at
`../../building-multimodal-rag/notebooks/food_tiny.db` and raises immediately if that
collection is empty, as a reminder to complete step 1 above first.

## How the test scenarios are built

1. Run the real pipeline (`CLIPEmbedder` → `MilvusDatabase` → `RAGProcessor`) for **Query A**,
   the on-topic steak question, giving a real `(answer_a, context_a)` pair.
2. Run the same real pipeline for **Query B**, an unrelated question about ice cream, giving a
   real `(answer_b, context_b)` pair.
3. Make one more real generation call — Query B's text, but with `context_a` (the steak
   images) as context — to get a genuinely off-topic answer (`answer_offtopic`) that's still a
   real model output, not a hardcoded string.
4. Combine these three real artifacts into the four scenarios, always judged against Query A:

   | Scenario | Answer | Context |
   | --- | --- | --- |
   | 1. Everything correct | `answer_a` | `context_a` |
   | 2. Off-topic answer | `answer_offtopic` | `context_a` |
   | 3. Wrong context | `answer_a` | `context_b` |
   | 4. Both degraded | `answer_b` | `context_b` |

No image is picked by hand and no answer text is invented — every scenario is a real
combination of two genuine pipeline runs.

### Using the pixi env as a kernel in VS Code / another IDE

`pixi run notebook` launches JupyterLab from inside the pixi env, so it already uses the right
kernel. If you instead open the notebook directly in an IDE (e.g. VS Code's Jupyter extension),
register the pixi env as a named ipykernel so it shows up in the kernel picker:

```bash
pixi run python -m ipykernel install --user --name three-rag-eval --display-name "Python (three-rag-eval pixi)"
```

Then in the notebook, select **"Python (three-rag-eval pixi)"** as the kernel (top-right kernel
picker in VS Code, or Kernel → Change Kernel in Jupyter). Re-select it any time the IDE seems to
be running against a stale environment.

## How it works

```
Query + Answer + Retrieved images
              │
              ▼
        RAGEvaluator          # Claude Sonnet 4.6 as LLM-as-a-Judge
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
Faithful-  Context   Answer
  ness     Rel.      Rel.
[0,1]     [0,1]     [0,1]
```

| Metric | What it measures | Inputs |
| --- | --- | --- |
| **Faithfulness** | Is the answer grounded in the retrieved images? | query + answer + images |
| **Context relevance** | Are the retrieved images useful for the query? | query + images |
| **Answer relevance** | Does the answer address the query? | query + answer |
