import argparse
import json
import os
from pathlib import Path

import gc
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from internal_representations.curvature_metrics import compute_trajectory


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_SOURCES = {
    "phi4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
    "deepseek-r1-qwen14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1-llama8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-llama70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwen3-14B": "Qwen/Qwen3-14B",
}


def compute_chunk_avg(H, k=500):
    """Average hidden states into non-overlapping chunks of k tokens (reasoning segments)."""
    chunked = []
    for i in range(0, H.size(1), k):
        chunked.append(H[:, i:i+k].mean(dim=1, keepdim=True))
    return torch.cat(chunked, dim=1)


def compute_segments_avg(x, boundaries):
    """Average hidden states within segments defined by boundary indices."""
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_avg = x[:, start:end].mean(dim=1)
        segments.append(segment_avg)
    return torch.stack(segments, dim=1)


def compute_lt_signals(averaged_hs):
    """Compute the three Latent-Trajectory (LT) signals from segment-level hidden states.

    Implements the three signals defined in Section 3.1 of the paper:
      - net_change:        Net Change      = ||h_N - h_1|| / N  (per layer)
      - cumulative_change: Cumulative Change = sum_n ||h_n - h_{n-1}||  (per layer)
      - aligned_change:    Aligned Change  = mean cosine similarity of step vectors
                           with the overall drift vector  (per layer)

    Args:
        averaged_hs: Tensor of shape [n_layers, n_segments, hidden_dim]

    Returns:
        dict mapping signal name to a list of per-layer scalar values
    """
    metrics = {
        "net_change": [],
        "cumulative_change": [],
        "aligned_change": [],
    }

    n_layers, N, _ = averaged_hs.shape

    for l in range(n_layers):
        H = averaged_hs[l].float()

        if N < 2:
            for key in metrics:
                metrics[key].append(float("nan"))
            continue

        # Drift vector: overall displacement from first to last segment
        h_first, h_last = H[0], H[-1]
        drift_vec = h_last - h_first
        net_norm = drift_vec.norm()

        # Update vectors: step-wise transitions between consecutive segments
        step_vecs = H[1:] - H[:-1]
        S = step_vecs.size(0)
        step_sizes = step_vecs.norm(dim=-1)

        # Net Change: magnitude of overall drift, normalised by number of segments
        metrics["net_change"].append((net_norm / S).item() if S > 0 else float("nan"))

        # Cumulative Change: total path length through representation space
        metrics["cumulative_change"].append(step_sizes.sum().item())

        # Aligned Change: mean cosine similarity of step vectors with the drift vector
        cos = F.cosine_similarity(step_vecs, drift_vec.unsqueeze(0), dim=-1)
        metrics["aligned_change"].append(cos.mean().item())

    return metrics


def compute_output_distribution_metrics(hs, lm_head):
    """Compute output-distribution-based confidence metrics (Section 4.1 baselines).

    Projects each segment's hidden state into vocabulary logit space and
    computes logit margin and entropy as confidence proxies.

    Args:
        hs:      Tensor of shape [n_layers, n_segments, hidden_dim]
        lm_head: Unembedding weight matrix [vocab_size, hidden_dim]

    Returns:
        dict with keys 'logit_margin_mean' and 'entropy_mean', each a list of per-layer values
    """
    metrics = {
        "logit_margin_mean": [],
        "entropy_mean": [],
    }

    lm_head = lm_head.to(hs.device)
    n_layers = hs.shape[0]

    for l in range(n_layers):
        H = hs[l]
        logits = torch.matmul(H, lm_head.T)  # [n_segments, vocab]

        top_k = logits.topk(2, dim=-1)[0]
        logit_margin_mean = (top_k[:, 0] - top_k[:, 1]).float().mean().item()

        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy_mean = entropy.float().mean().item()

        metrics["logit_margin_mean"].append(logit_margin_mean)
        metrics["entropy_mean"].append(entropy_mean)

    return metrics


def build_llm_input(model_name, prompt, think_ans, tokenizer):
    """Construct the full input string (prompt + thinking trace) for a given model."""
    if model_name.startswith("phi"):
        system_msg = (
            "You are Phi, a language model trained by Microsoft to help users. "
            "Your role as an assistant involves thoroughly exploring questions through a "
            "systematic thinking process before providing the final precise and accurate "
            "solutions. This requires engaging in a comprehensive cycle of analysis, "
            "summarizing, exploration, reassessment, reflection, backtracing, and iteration "
            "to develop well-considered thinking process. Please structure your response into "
            "two main sections: Thought and Solution using the specified format: "
            "<think> {Thought section} </think> {Solution section}. In the Thought section, "
            "detail your reasoning process in steps. Each step should include detailed "
            "considerations such as analysing questions, summarizing relevant findings, "
            "brainstorming new ideas, verifying the accuracy of the current steps, refining "
            "any errors, and revisiting previous steps. In the Solution section, based on "
            "various attempts, explorations, and reflections from the Thought section, "
            "systematically present the final solution that you deem correct. The Solution "
            "section should be logical, accurate, and concise and detail necessary steps "
            "needed to reach the conclusion. Now, try to solve the following question "
            "through the above guidelines:"
        )
        chat = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt_str, f"{prompt_str}{think_ans}"

    elif model_name.startswith("deepseek"):
        # prompt_length must match old v3: len(tokenize(raw_prompt)) only,
        # so <think>\n is treated as part of the thinking region, not the prompt.
        return prompt, f"{prompt}<think>\n{think_ans}"

    elif model_name.startswith("qwen"):
        chat = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        return prompt_str, f"{prompt_str}{think_ans}"

    else:
        raise ValueError(f"Model {model_name} not recognised.")


def _load_eureka_data(path, dataset, model_name):
    """Load Eureka-format JSONL reports and return a DataFrame."""
    report = Path(path) / f"results/eureka_reports/{dataset}/{model_name}.jsonl"
    with open(report, "r", encoding="utf-8") as f:
        data = pd.DataFrame([json.loads(line) for line in f])

    data = data.loc[data["data_repeat_id"].isin([f"repeat_{i}" for i in range(5)])]

    if dataset == "TSP":
        ids = (
            data.drop_duplicates("data_point_id")[["data_point_id", "category"]]
            .groupby("category", group_keys=False)
            .apply(lambda g: g.sample(n=20, random_state=0))["data_point_id"]
        )
        data = data[data["data_point_id"].isin(ids)]
    elif (dataset == "GPQA") and (model_name == "deepseek-r1-qwen32B"):
        data["data_point_id"] = data["Record ID"].astype("category").cat.codes
        data = (
            data.sort_values("response_time", ascending=True)
            .drop_duplicates(subset=["data_point_id", "data_repeat_id"])
            .reset_index(drop=True)
        )

    return data


def compute_internals(
    model_name: str,
    path: str,
    dataset: str | None = None,
    results_file: str | None = None,
    average_type: str = "ntokens",
    n_tokens: int = 500,
    cache_dir: str | None = None,
    cuda_devices: str = "0",
):
    """Compute LT signals and baseline metrics from model hidden states.

    Supports two data sources:
      - Eureka JSONL reports (``--dataset``): used for GPQA, AIME2025, and
        phi4-reasoning-plus TSP results.
      - Pre-generated JSON files (``--results_file``): used for TSP and BigBench
        results produced by generate_tsp.py / generate_bigbench.py, where the
        thinking trace is already split out as ``think_answer``.
    """
    if dataset is None and results_file is None:
        raise ValueError("Provide either --dataset (Eureka) or --results_file (custom JSON).")

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Model '{model_name}' not recognised. Choose from: {list(MODEL_SOURCES)}")
    model_source = MODEL_SOURCES[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    lm_head = model.lm_head.weight.detach()
    model.eval()

    use_custom_json = results_file is not None
    if results_file is not None:
        results_file_path = Path(results_file)
        with open(results_file_path, "r", encoding="utf-8") as f:
            data = pd.DataFrame(json.load(f))
        output_subdir: str | Path = results_file_path.parent.relative_to(Path(path) / "results")
    else:
        assert dataset is not None
        data = _load_eureka_data(path, dataset, model_name)
        output_subdir = dataset

    results = []
    for _, i_data in tqdm(enumerate(data.itertuples()), total=len(data)):

        prompt = i_data.prompt

        if use_custom_json:
            think_ans_raw = i_data.think_answer
            if think_ans_raw is None or not isinstance(think_ans_raw, str):
                print(f"Skipping data point {i_data.data_point_id}: think_answer is None.")
                continue
            think_ans = think_ans_raw + "</think>"
            llm_input = f"{prompt}{think_ans}"
            prompt_length = len(tokenizer.encode(prompt))
        else:
            try:
                ans = i_data.raw_model_output
            except AttributeError:
                ans = i_data.model_output
            if not isinstance(ans, str):
                print(f"Skipping data point {i_data.data_point_id}: model output is None.")
                continue
            think_ans = ans.split("</think>")[0] + "</think>"
            prompt_str, llm_input = build_llm_input(model_name, prompt, think_ans, tokenizer)
            prompt_length = len(tokenizer.encode(prompt_str))

        n_think_tokens = len(tokenizer.encode(think_ans))

        results_entry = {
            "data_point_id": i_data.data_point_id,
            "data_repeat_id": i_data.data_repeat_id,
            "n_think_tokens": n_think_tokens,
        }
        if use_custom_json:
            results_entry["accuracy"] = i_data.accuracy

        encoding = tokenizer(
            llm_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                output_hidden_states=True,
            )

        # Stack hidden states; each entry in outputs.hidden_states is [batch, seq_len, hidden_dim].
        # After stacking and permuting: [n_layers, seq_len, hidden_dim] (batch_size=1 squeezed out).
        hs = torch.stack(outputs.hidden_states[1:]).permute(1, 0, 2, 3)  # [batch, n_layers, seq_len, hidden_dim]
        del outputs
        hs = hs.squeeze(0)[:, prompt_length:, :]  # [n_layers, think_len, hidden_dim]

        # Compute output-distribution baselines
        dist_metrics = compute_output_distribution_metrics(hs, lm_head=lm_head)
        results_entry.update(dist_metrics)
        del dist_metrics

        # Average hidden states into reasoning segments
        if average_type == "ntokens":
            averaged_hs = compute_chunk_avg(hs, k=n_tokens)
        elif average_type == "segments":
            new_line_idxs = [
                idx for idx, t in enumerate(encoding["input_ids"][0, prompt_length:])
                if "\n" in tokenizer.decode(t)
            ]
            averaged_hs = compute_segments_avg(hs, boundaries=new_line_idxs)
        else:
            raise ValueError(f"Average type '{average_type}' not recognised.")

        del encoding, hs
        torch.cuda.empty_cache()
        gc.collect()

        # Compute LT signals (Section 3.1)
        lt_metrics = compute_lt_signals(averaged_hs)
        results_entry.update(lt_metrics)
        del lt_metrics

        # Compute cross-layer signals (Section 4.1 baselines)
        cross_layer = compute_trajectory(averaged_hs, traj_dim=0)
        results_entry.update(cross_layer)
        del cross_layer

        # Compute sequence-wise trajectory signals
        seq_traj = compute_trajectory(averaged_hs, traj_dim=1)
        results_entry.update(seq_traj)
        del seq_traj

        del averaged_hs
        torch.cuda.empty_cache()
        gc.collect()

        results.append(results_entry)

    # Save results
    avg_tag = f"ntokens_{n_tokens}" if average_type == "ntokens" else average_type
    output_file = Path(path) / "results" / "internals" / output_subdir / f"{model_name}_{avg_tag}_internals.pt"
    os.makedirs(output_file.parent, exist_ok=True)
    torch.save(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Latent-Trajectory signals from model hidden states.")
    parser.add_argument(
        "--model",
        type=str,
        default="phi4-reasoning-plus",
        choices=list(MODEL_SOURCES),
        help="Reasoning model to analyse.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        type=str,
        choices=["GPQA", "AIME2025", "TSP"],
        help="Eureka dataset name (loads from results/eureka_reports/<dataset>/<model>.jsonl).",
    )
    source.add_argument(
        "--results_file",
        type=str,
        help=(
            "Path to a pre-generated JSON results file produced by generate_tsp.py or "
            "generate_bigbench.py. Output directory mirrors the results file location under "
            "results/internals/."
        ),
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the project root directory.",
    )
    parser.add_argument(
        "--average_type",
        type=str,
        default="ntokens",
        choices=["ntokens", "segments"],
        help="How to aggregate token-level hidden states into reasoning segments.",
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=500,
        help="Tokens per segment when --average_type=ntokens (default: 500).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching downloaded model weights.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default="0",
        help="Comma-separated list of CUDA device indices, e.g. '0,1'.",
    )
    args = parser.parse_args()

    compute_internals(
        model_name=args.model,
        path=args.path,
        dataset=args.dataset,
        results_file=args.results_file,
        average_type=args.average_type,
        n_tokens=args.n_tokens,
        cache_dir=args.cache_dir,
        cuda_devices=args.cuda_devices,
    )
