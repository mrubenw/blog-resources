# Creating JEV-like Typed Decisions

Constraining an LLM to a fixed, typed set of answer options by reading logits over a
small "verbalizer" vocabulary (e.g. single-letter choices `A`–`Z`), instead of parsing
free-form generated text.

## Setup

```bash
# Install pixi: https://pixi.sh/latest/#installation
pixi install
```

No API key is needed — the notebook loads `Qwen/Qwen3.5-4B` locally via `transformers`.

## Run

```bash
pixi run notebook   # JupyterLab
```

Open `notebooks/typed_decisions.ipynb`.

### Using the pixi env as a kernel in VS Code / another IDE

`pixi run notebook` launches JupyterLab from inside the pixi env, so it already uses the right
kernel. If you instead open the notebook directly in an IDE (e.g. VS Code's Jupyter extension),
register the pixi env as a named ipykernel so it shows up in the kernel picker:

```bash
pixi run python -m ipykernel install --user --name typed-decisions --display-name "Python (typed-decisions pixi)"
```

Then in the notebook, select **"Python (typed-decisions pixi)"** as the kernel (top-right kernel
picker in VS Code, or Kernel → Change Kernel in Jupyter).

## How it works

1. Load the model and tokenizer, and resolve each verbalizer option (e.g. `A`, `B`, `C`, ...)
   to its single token id.
2. Build a prompt that asks the model to pick one of the typed options.
3. Run a forward pass and read the logits at the final position.
4. Restrict the logits to just the verbalizer token ids and softmax over them, turning the
   full-vocabulary distribution into a probability over a small, typed set of answers.
5. Wrap the result into a typed answer (option + confidence) instead of parsing free text.
