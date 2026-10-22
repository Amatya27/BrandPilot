# BrandPilot

[![CI](https://github.com/Amatya27/BrandPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Amatya27/BrandPilot/actions/workflows/ci.yml)

BrandPilot researches a company's marketing from the public web and drafts a collaboration email for it. You give it a company name (and, if you have one, a person or role at that company). It searches the web, pulls the useful passages, works out who the brand is talking to and where its marketing has gaps, and writes an outreach email plus an Instagram post idea. Every gap comes with the source it was based on, so the result can be checked before anyone sends it.

I started it as a GenAI course project and adapted it for the Center for Entrepreneurship at W. P. Carey, where we reach out to companies about collaborations. Researching one company by hand used to take about 45 minutes. With BrandPilot it takes under 10: staff read the brief, check the cited sources, edit the draft and send it.

## How it works

Two stages, kept separate on purpose:

1. **Retrieval (RAG).** `ddgs` runs a set of web searches for the company. Pages are fetched by 4 workers in parallel, cleaned, cached, and split into 1,200-character chunks with 200 characters of overlap. Chunks are scored for how much they say about the company and its marketing, and the best few go into the prompt with their source links.
2. **Generation (LoRA).** A Gemma 3 1B model with a rank-16 LoRA adapter turns that evidence into a strict JSON brief. I fine-tuned it on a custom dataset of 120 examples (2.6M trainable parameters, about 0.26% of the model). Held-out evaluation loss went from 1.23 to 0.87.

Keeping facts in retrieval and style in the fine-tune means the model does not have to remember anything about the company. It only has to write well from what it is given.

The model's output is never trusted as-is. `brand_assistant_schema.py` pulls the JSON object out of whatever the model returns (code fences, extra text around it), checks every required field, and the post-processing step fills or repairs weak fields from the evidence before anything reaches the screen.

The web app is a FastAPI service. A request starts a background job on a 2-worker pool, the page polls the job for progress, and `/api/health` reports status.

```
company name ──> web search ──> fetch + chunk + score ──> evidence packet
                                                               │
                         browser <── FastAPI <── validate <── Gemma 3 1B + LoRA
```

## Output

The model returns JSON with these fields:

`brand_summary`, `likely_target_audience`, `current_marketing_strategy`, `marketing_gaps`, `instagram_post_idea` (hook, concept, why it closes the gap), `email_subject`, `email_body`, `citations`.

I moved from free text to JSON because free text was hard to validate and the early versions kept getting cut off. The notebook generated at 512 tokens in total while the prompt alone used most of that. A compact evidence block, JSON output and a separate token budget for the answer fixed it.

## Running it

### Option 1: Docker (app in a container, model on the Mac)

The app and the retrieval pipeline run in Docker. The model runs on the Mac with MLX, because MLX uses Apple's GPU and Docker containers on a Mac cannot. The container talks to the model server at `host.docker.internal:8080`.

```bash
# terminal 1: model server on the Mac
mlx_lm.server --model mlx-community/gemma-3-1b-it-qat-4bit --port 8080

# terminal 2: the app
docker compose up --build
```

Open http://localhost:8010. Results are saved to `./outputs`. Stop with `docker compose down`.

If the container cannot reach the model, start the server with `--host 0.0.0.0`.

A few choices in the Docker setup worth knowing:
- dependencies install in their own layer, so changing code does not reinstall everything
- the app runs as a non-root user
- a health check hits `/api/health`
- `.dockerignore` keeps model weights, adapters, caches and credentials such as `kaggle.json` out of the image

### Option 2: Local Python

```bash
conda create -n lora-brand-assistant python=3.11 -y
conda activate lora-brand-assistant
pip install -r requirements-local.txt -r requirements-app.txt
```

Retrieval only, no model needed (a good first check):

```bash
python run_brand_assistant.py --backend rag-only --company "Airbnb"
```

With the MLX model server running (see above):

```bash
python run_brand_assistant.py \
  --backend openai-compatible \
  --api-base http://localhost:8080/v1 \
  --model mlx-community/gemma-3-1b-it-qat-4bit \
  --company "Airbnb" --person "Director of Brand Marketing"
```

Web app: `./run_app.sh`, then open http://127.0.0.1:8010.

### Using the original Keras checkpoint

The first version of the adapter was trained in Keras on Colab, and the code can still load it (`--backend keras`, needs `requirements-keras.txt` and Kaggle credentials for the Gemma preset). The checkpoint files are not in this repo. For a Mac, MLX is the faster path.

### Training the adapter with MLX

```bash
python prepare_mlx_dataset.py --input path/to/brand_outreach_clean_v2.jsonl --output-dir data/mlx_brand_assistant

mlx_lm.lora --model mlx-community/gemma-3-1b-it-qat-4bit --train \
  --data data/mlx_brand_assistant --iters 600 --batch-size 1 \
  --learning-rate 1e-5 --mask-prompt --adapter-path adapters/brand_assistant_1b_qat
```

`--mask-prompt` computes the loss only on the answer, not the prompt. Pass the same `--adapter-path` to `mlx_lm.server` to serve the fine-tuned model, and start the server from the folder that path is relative to.

## Tests and CI

```bash
pip install -r requirements-app.txt -r requirements-local.txt -r requirements-dev.txt
pytest -q
```

The 17 tests cover the parts that should behave the same every time: pulling JSON out of messy model output and validating it, chunking and company matching in retrieval, and the API (health check, rejecting a blank company, 404 for an unknown job). They need no model and no network.

GitHub Actions runs on every push: it installs dependencies, runs the tests, builds the Docker image, starts the container and checks that `/api/health` answers.

## Project layout

| File | What it does |
|---|---|
| `app.py` | FastAPI server and job handling |
| `brand_assistant_service.py` | ties retrieval, generation and post-processing together |
| `brand_assistant_rag.py` | search, fetching, chunking, scoring, prompt building |
| `brand_assistant_schema.py` | JSON extraction and validation |
| `brand_assistant_postprocess.py` | cleans up the brief and links each point to sources |
| `brand_assistant_openai_backend.py` | calls an OpenAI-compatible server (MLX) |
| `brand_assistant_keras_backend.py`, `brand_assistant_checkpoint.py` | Keras checkpoint loading and inference |
| `prepare_mlx_dataset.py` | converts the training data to MLX format |
| `run_brand_assistant.py` | command-line entry point |
| `webapp/` | browser front end |
| `tests/` | pytest suite |

## What I would do next

- Measure retrieval properly: a small set of companies with hand-checked sources, and track how often the right pages make it into the prompt.
- Retrain the adapter in MLX on a larger dataset and compare outputs side by side with the Keras version.
- Add request metrics (latency per stage) so slow searches are easy to spot.
