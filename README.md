# VisionXAI-ModelComparison

## Project Building and Running (Docker)

This repository provides notebooks, model wrappers, and an evaluation script to compare image-captioning models (InceptionV3 vs EfficientNetB4) using a shared tokenizer and deterministic decoding.

### Build image

Use a descriptive image tag so you can run multiple builds without confusion. Example tag used here:

```cmd
cd environment && docker build . --tag visionxai-modelcomparison:20251216
```

### Run image (recommended)

Run the container with mounted `data`, `code`, and `results` directories so model artifacts and outputs persist on the host. The examples below mount the current working directory's `data`, `code`, and `results` into the container.

Windows (PowerShell):

```powershell
docker run --platform linux/amd64 --rm --gpus all --workdir /code `
  --volume "${PWD}/data":/data `
  --volume "${PWD}/code":/code `
  --volume "${PWD}/results":/results `
  visionxai-modelcomparison:20251216 /bin/bash
```

Windows (Command Prompt):

```cmd
docker run --platform linux/amd64 --rm --gpus all --workdir /code ^
  --volume "%CD%/data":/data ^
  --volume "%CD%/code":/code ^
  --volume "%CD%/results":/results ^
  visionxai-modelcomparison:20251216 /bin/bash
```

Linux / macOS:

```bash
docker run --platform linux/amd64 --rm --gpus all --workdir /code \
  -v "$(pwd)/data:/data" -v "$(pwd)/code:/code" -v "$(pwd)/results:/results" \
  visionxai-modelcomparison:20251216 /bin/bash
```

Once inside the container, the Python virtual environment is activated by default because `/opt/venv/bin` is in the PATH. If you need to activate it manually:

```bash
source /opt/venv/bin/activate
```

### Standardized evaluation workflow

You can run the complete evaluation for both models in a single step using the master script `code/run`:

Windows (PowerShell):

```powershell
docker run --platform linux/amd64 --rm --gpus all --workdir /code `
  --volume "${PWD}/data":/data `
  --volume "${PWD}/code":/code `
  --volume "${PWD}/results":/results `
  visionxai-modelcomparison:20251216 ./run
```

Windows (Command Prompt):

```cmd
docker run --platform linux/amd64 --rm --gpus all --workdir /code ^
  --volume "%CD%/data":/data ^
  --volume "%CD%/code":/code ^
  --volume "%CD%/results":/results ^
  visionxai-modelcomparison:20251216 ./run
```

Linux / macOS:

```bash
docker run --platform linux/amd64 --rm --gpus all --workdir /code \
  -v "$(pwd)/data:/data" -v "$(pwd)/code:/code" -v "$(pwd)/results:/results" \
  visionxai-modelcomparison:20251216 ./run
```

Alternatively, you can run individual evaluations or customize the parameters as shown below.

- Ensure both models use their corresponding tokenizer/vocabulary files, or a shared tokenizer when appropriate.
- When running evaluation inside the container (which mounts the `code` directory to `/code`), the script `evaluate_captions.py` is executed from the working directory `/code`, and the import paths do not require the `code.` prefix.
- The datasets, models, and results are accessed via `/data` and `/results` mounts.

Example: Evaluate InceptionV3 inside Docker

```bash
docker run --platform linux/amd64 --rm --gpus all --workdir /code \
  --volume "$(pwd)/data":/data \
  --volume "$(pwd)/code":/code \
  --volume "$(pwd)/results":/results \
  visionxai-modelcomparison:20251216 \
  python evaluate_captions.py \
    --model-module InceptionV3.model_factory:build_model \
    --checkpoints-dir /data/InceptionV3/best_model \
    --val-annotations /data/InceptionV3/test/BNATURE/caption/validation.txt \
    --test-annotations /data/InceptionV3/test/BNATURE/caption/test.txt \
    --vocab /data/InceptionV3/tokenizer.pkl \
    --decode greedy --seed 1234 --output-dir /results/eval_inception
```

Example: Evaluate EfficientNetB4 inside Docker

```bash
docker run --platform linux/amd64 --rm --gpus all --workdir /code \
  --volume "$(pwd)/data":/data \
  --volume "$(pwd)/code":/code \
  --volume "$(pwd)/results":/results \
  visionxai-modelcomparison:20251216 \
  python evaluate_captions.py \
    --model-module EfficientNetB4.model_factory:build_model \
    --checkpoints-dir /data/EfficientNetB4/Model_weights/20250625_042759/Temp \
    --val-annotations /data/InceptionV3/test/BNATURE/caption/validation.txt \
    --test-annotations /data/InceptionV3/test/BNATURE/caption/test.txt \
    --vocab /data/EfficientNetB4/Vocab/20250625_042759/vocab_20250625_042759 \
    --decode greedy --seed 1234 --output-dir /results/eval_efficientnet
```

### Quick troubleshooting

- If evaluation fails to restore a TF checkpoint, generate `predictions.json` from the notebook and re-run the evaluator pointing to the folder containing `predictions.json`.
- Confirm the `--vocab` path points to the tokenizer/vocab file used to train both models (same tokenizer for fair comparison).
- Set `--limit <N>` to run evaluation on a smaller subset of images (e.g. `--limit 10`) for fast debugging and verification.

### Reproducibility

Use the same `--vocab` and `--seed` for both model runs to ensure deterministic, comparable results.
