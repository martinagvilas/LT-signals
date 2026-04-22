import argparse
import gc
import json
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


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
        chunked.append(H[:, i:i + k].mean(dim=1, keepdim=True))
    return torch.cat(chunked, dim=1)


def compute_step_metrics(averaged_hs):
    """Compute per-step LT signals averaged over layers.

    Args:
        averaged_hs: Tensor of shape [n_layers, n_segments, hidden_dim]

    Returns:
        dict with keys:
          net_by_step:        [n_segments-1] — per-step net displacement from the
                              first segment, averaged over layers
          cumulative_by_step: [n_segments-1] — per-step consecutive displacement,
                              averaged over layers
    """
    # net displacement from first segment at each step: [n_layers, n_steps, hidden_dim]
    net_diff = averaged_hs[:, 1:, :] - averaged_hs[:, 0:1, :]
    net_by_step = torch.norm(net_diff, dim=-1).mean(dim=0)  # [n_steps]

    # consecutive step displacement: [n_layers, n_steps, hidden_dim]
    consec_diff = averaged_hs[:, 1:, :] - averaged_hs[:, :-1, :]
    cumulative_by_step = torch.norm(consec_diff, dim=-1).mean(dim=0)  # [n_steps]

    return {
        "net_by_step": net_by_step.cpu(),
        "cumulative_by_step": cumulative_by_step.cpu(),
    }


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
    elif dataset == "GPQA" and model_name == "deepseek-r1-qwen32B":
        data["data_point_id"] = data["Record ID"].astype("category").cat.codes
        data = (
            data.sort_values("response_time", ascending=True)
            .drop_duplicates(subset=["data_point_id", "data_repeat_id"])
            .reset_index(drop=True)
        )

    return data


def compute_step_vectors(
    model_name,
    dataset,
    path,
    results_file=None,
    n_tokens=500,
    cache_dir=None,
    cuda_devices="0",
):
    """Compute per-step LT signals from model hidden states.

    Supports two data sources:
      - Eureka JSONL reports (``--dataset``): GPQA, AIME2025, and phi4-reasoning-plus TSP.
      - Pre-generated JSON files (``--results_file``): TSP for non-phi4 models produced
        by generate_tsp.py.

    Produces a list of dicts saved to:
      results/internals/{dataset}/{model_name}_ntokens_{n_tokens}_step_vectors_residual.pt

    Each dict has keys:
      data_point_id, data_repeat_id, n_think_tokens,
      net_by_step, cumulative_by_step
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Model '{model_name}' not in {list(MODEL_SOURCES)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCES[model_name], trust_remote_code=True, cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SOURCES[model_name],
        device_map="auto",
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.eval()

    use_custom_json = results_file is not None
    if results_file is not None:
        with open(results_file, "r", encoding="utf-8") as f:
            data = pd.DataFrame(json.load(f))
    else:
        data = _load_eureka_data(path, dataset, model_name)

    results = []
    for _, row in tqdm(enumerate(data.itertuples()), total=len(data)):
        if use_custom_json:
            think_ans_raw = row.think_answer
            if not isinstance(think_ans_raw, str):
                print(f"Skipping {row.data_point_id}: think_answer is None.")
                continue
            think_ans = think_ans_raw + "</think>"
            llm_input = f"{row.prompt}{think_ans}"
            prompt_length = len(tokenizer.encode(row.prompt))
        else:
            try:
                ans = row.raw_model_output
            except AttributeError:
                ans = row.model_output
            if not isinstance(ans, str):
                print(f"Skipping {row.data_point_id}: model output is None.")
                continue
            think_ans = ans.split("</think>")[0] + "</think>"
            prompt_str, llm_input = build_llm_input(model_name, row.prompt, think_ans, tokenizer)
            prompt_length = len(tokenizer.encode(prompt_str))

        n_think_tokens = len(tokenizer.encode(think_ans))

        encoding = tokenizer(
            llm_input, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                output_hidden_states=True,
            )

        # [n_layers, think_len, hidden_dim]
        hs = torch.stack(outputs.hidden_states[1:]).permute(1, 0, 2, 3).squeeze(0)
        hs = hs[:, prompt_length:, :]
        del outputs, encoding
        torch.cuda.empty_cache()
        gc.collect()

        averaged_hs = compute_chunk_avg(hs, k=n_tokens).float()
        del hs

        entry = {
            "data_point_id":  row.data_point_id,
            "data_repeat_id": row.data_repeat_id,
            "n_think_tokens": n_think_tokens,
        }
        entry.update(compute_step_metrics(averaged_hs))
        del averaged_hs
        torch.cuda.empty_cache()
        gc.collect()

        results.append(entry)

    output_file = (
        Path(path) / "results" / "internals" / dataset
        / f"{model_name}_ntokens_{n_tokens}_step_vectors_residual.pt"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_file)
    print(f"Saved {len(results)} entries to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-step LT signals from model hidden states."
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=list(MODEL_SOURCES),
        help="Reasoning model to analyse.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["GPQA", "AIME2025", "TSP"],
        help="Dataset name. Used to determine the output path.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--results_file", type=str, default=None,
        help=(
            "Path to a pre-generated JSON results file produced by generate_tsp.py. "
            "When omitted, loads from results/eureka_reports/<dataset>/<model>.jsonl."
        ),
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the project root directory.",
    )
    parser.add_argument(
        "--n_tokens", type=int, default=500,
        help="Tokens per reasoning segment (default: 500).",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory for caching downloaded model weights.",
    )
    parser.add_argument(
        "--cuda_devices", type=str, default="0",
        help="Comma-separated CUDA device indices, e.g. '0,1'.",
    )
    args = parser.parse_args()

    compute_step_vectors(
        model_name=args.model,
        dataset=args.dataset,
        path=args.path,
        results_file=args.results_file,
        n_tokens=args.n_tokens,
        cache_dir=args.cache_dir,
        cuda_devices=args.cuda_devices,
    )
