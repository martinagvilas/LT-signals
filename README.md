# TRACING THE TRACES: LATENT TEMPORAL SIGNALS FOR EFFICIENT AND ACCURATE REASONING

Code accompanying the paper https://arxiv.org/abs/2510.10494, published at ICLR 2026.

## Environment setup

The pipeline requires Python 3.10, PyTorch with CUDA 12, vLLM (for generation), and HuggingFace Transformers (for hidden-state extraction). Create a conda environment from scratch:

```bash
conda create -n lt-signals python=3.10 -y
conda activate lt-signals

# Core dependencies (exact versions used for the paper)
pip install -r requirements.txt

# Install this package (makes src/ importable by the compute scripts)
pip install -e .
```

## Reproducing paper results

The pipeline has three stages: **generation → internal representations → analysis**.
Stages 1 and 2 require GPU access. Stage 3 (notebooks) runs on CPU.

### 1. Generate model outputs

This step generates and stores the full reasoning trace.

**GPQA and AIME2025** outputs were generated via the [Eureka](https://github.com/ServiceNow/eureka-ml-insights)
evaluation framework. The resulting JSONL report files are expected at:

```
results/eureka_reports/{dataset}/{model_name}.jsonl
```

**TSP and BigBench** outputs are generated with the scripts in `src/generation/`:

```bash
# TSP — phi4-reasoning-plus uses the Eureka report directly; run this for all other models
python src/generation/generate_tsp.py \
    --model deepseek-r1-qwen14B \
    --path . --cuda_devices 0,1

# BigBench (fables | social-iqa | codenames | com2sense)
python src/generation/generate_bigbench.py \
    --model phi4-reasoning-plus --task fables \
    --benchmarks_path /path/to/BIG-bench \
    --path . --cuda_devices 0,1
```

TSP output: `results/TSP/{model_name}_results.json`
BigBench output: `results/bigbench/{task}/{model_name}_results.json`

Repeat for each model (`phi4-reasoning-plus`, `deepseek-r1-qwen14B`, `deepseek-r1-qwen32B`,
`deepseek-r1-llama8B`, `deepseek-r1-llama70B`, `qwen3-14B`).

### 2. Extract hidden states and compute internal representations

All three scripts load a model, run a forward pass over each reasoning trace, and save the
resulting representations to disk. Run them for each model/dataset combination.

#### LT signals and Wang et al. baselines

For GPQA and AIME2025 (Eureka JSONL reports):

```bash
python src/internal_representations/compute_internal_representations.py \
    --model phi4-reasoning-plus --dataset GPQA \
    --n_tokens 500 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1

python src/internal_representations/compute_internal_representations.py \
    --model phi4-reasoning-plus --dataset AIME2025 \
    --n_tokens 500 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1
```

For TSP (non-phi4 models) and BigBench (custom JSON results files from stage 1):

```bash
# TSP — non-phi4 models only (phi4 uses --dataset TSP above)
python src/internal_representations/compute_internal_representations.py \
    --model deepseek-r1-qwen14B \
    --results_file results/TSP/deepseek-r1-qwen14B_results.json \
    --n_tokens 500 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1

# BigBench
python src/internal_representations/compute_internal_representations.py \
    --model phi4-reasoning-plus \
    --results_file results/bigbench/fables/phi4-reasoning-plus_results.json \
    --n_tokens 500 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1
```

Repeat for each model and dataset.
The `--n_tokens` argument controls the segment length; the paper uses 500 by default
(also run with 100, 300, 700 to reproduce the segment-length ablation).

Output: `results/internals/{dataset}/{model_name}_ntokens_{n_tokens}_internals.pt`

#### Early-exit output-distribution baselines

Computes logit margin, entropy, and perplexity at each percentage of the thinking trace.

```bash
python src/output_distribution/compute_early_exit_properties.py \
    --model phi4-reasoning-plus --dataset GPQA \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1

python src/output_distribution/compute_early_exit_properties.py \
    --model phi4-reasoning-plus --dataset AIME2025 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1

python src/output_distribution/compute_early_exit_properties.py \
    --model phi4-reasoning-plus --dataset TSP \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1
```

Repeat for `deepseek-r1-qwen14B` and `qwen3-14B` on GPQA and AIME2025. For TSP,
`deepseek-r1-qwen14B` and `qwen3-14B` use the custom JSON from stage 1:

```bash
python src/output_distribution/compute_early_exit_properties.py \
    --model deepseek-r1-qwen14B --dataset TSP \
    --results_file results/TSP/deepseek-r1-qwen14B_results.json \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1
```

Output: `results/early_exit/{dataset}/{model_name}_output_properties.json`

#### Per-step vectors

Computes LT signals at each reasoning step (500-token segment) rather than as a single
aggregate.

```bash
python src/internal_representations/compute_step_vectors.py \
    --model phi4-reasoning-plus --dataset GPQA \
    --n_tokens 500 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1

python src/internal_representations/compute_step_vectors.py \
    --model deepseek-r1-qwen14B --dataset TSP \
    --results_file results/TSP/deepseek-r1-qwen14B_results.json \
    --n_tokens 500 \
    --path . --cache_dir /path/to/hf/cache --cuda_devices 0,1
```

Output: `results/internals/{dataset}/{model_name}_ntokens_500_step_vectors_residual.pt`

### 3. Run the analysis notebooks

Open the notebooks in `notebooks/` with JupyterLab (`jupyter lab`) and run them in order.
Each notebook sets `PATH` to the project root — update cell 2 if your working directory differs.

| Notebook | Paper section | Requires |
|---|---|---|
| `LT_predictivity.ipynb` | Section 5.1 — ROC-AUC of all metrics | `*_internals.pt`, `*_output_properties.json` |
| `multi_sample_experiments.ipynb` | Section 5.2 — sequential LT sampling, Table 1 | `*_internals.pt`, `*_output_properties.json` |
| `early_sample_experiments.ipynb` | Section 5.3 — classifier-based sample selection | `*_step_vectors_residual.pt`, `*_output_properties.json` |
