# blog-resources

Code for all published articles. Each article lives under `articles/` as a self-contained project with its own environment managed by [pixi](https://pixi.sh).

## Articles

| Article | Directory | Published |
|---------|-----------|-----------|
| Building a Multimodal RAG Pipeline | [articles/building-multimodal-rag](articles/building-multimodal-rag) | — |

## Structure

```
blog-resources/
├── articles/
│   └── <article-slug>/
│       ├── README.md
│       ├── pixi.toml          # per-article env
│       ├── pyproject.toml     # package definition
│       ├── .env.example
│       ├── notebooks/
│       ├── src/
│       ├── tests/
│       └── data/              # gitignored, fetched by script
```

## Running an article

```bash
cd articles/<article-slug>
pixi install
pixi run test          # run unit tests
pixi run notebook      # launch JupyterLab
```
