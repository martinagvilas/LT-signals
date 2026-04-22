"""Compute output-distribution confidence baselines at early-exit percentages (Section 4.2).

For each sample, truncates the thinking trace at 0%, 10%, ..., 100% of its full
token length, appends a forced-answer suffix, and generates up to n_new_tokens
new tokens. Records for each percentage p:

  - acc_{p}:             accuracy (ground truth contained in output)
  - think_n_tokens_{p}: number of thinking tokens used
  - entropy_{p}:        per-layer entropy of the first new token  [list, n_layers]
  - logit_margin_{p}:   per-layer top-2 logit margin              [list, n_layers]
  - perplexity_{p}:     per-layer inverse top-1 probability       [list, n_layers]

Hidden-state projections follow the logit-lens approach:
model.model.norm(hs) → model.lm_head → softmax.

Usage:
    python compute_early_exit_properties.py --model phi4-reasoning-plus \\
        --dataset GPQA --path /path/to/project --cuda_devices 0,1
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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

# Short answer: generate 2 tokens (letter/number). Math/TSP: generate 8 tokens.
N_NEW_TOKENS = {"GPQA": 2, "AIME2025": 8, "TSP": 8}

PHI_SYSTEM_MSG = (
    "You are Phi, a language model trained by Microsoft to help users. "
    "Your role as an assistant involves thoroughly exploring questions through a "
    "systematic thinking process before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, "
    "reassessment, reflection, backtracing, and iteration to develop well-considered thinking "
    "process. Please structure your response into two main sections: Thought and Solution using "
    "the specified format: <think> {Thought section} </think> {Solution section}. In the Thought "
    "section, detail your reasoning process in steps. Each step should include detailed "
    "considerations such as analysing questions, summarizing relevant findings, brainstorming new "
    "ideas, verifying the accuracy of the current steps, refining any errors, and revisiting "
    "previous steps. In the Solution section, based on various attempts, explorations, and "
    "reflections from the Thought section, systematically present the final solution that you "
    "deem correct. The Solution section should be logical, accurate, and concise and detail "
    "necessary steps needed to reach the conclusion. Now, try to solve the following question "
    "through the above guidelines:"
)

# Forced-answer suffix appended after the truncated thinking trace.
# Each suffix ends the thinking section and starts the answer.
EARLY_EXIT_SUFFIX = {
    "phi": {
        "GPQA":    "I'll produce final answer</think>\n Final answer:",
        "AIME2025": "I'll produce final answer</think>\nFinal answer: ",
        "TSP":     "I'll produce final answer</think>\n'Total distance': ",
    },
    "deepseek": {
        "GPQA":    "\n\n</think>\nFinal answer: \\boxed{",
        "AIME2025": "\n\n</think>\nFinal answer: \\boxed{",
        "TSP":     "\n\n</think>\n'Total distance': ",
    },
    "qwen": {
        "GPQA":    "\n</think>\n\nFinal answer:",
        "AIME2025": "\n</think>\n\nFinal answer:",
        "TSP":     "\n\n</think>\n'TotalDistance': ",
    },
}


def _get_suffix(model_name, dataset):
    for prefix, suffixes in EARLY_EXIT_SUFFIX.items():
        if model_name.startswith(prefix):
            return suffixes[dataset]
    raise ValueError(f"No early-exit suffix defined for model={model_name}, dataset={dataset}")


def _format_prompt(model_name, raw_prompt, tokenizer):
    """Apply model-specific chat template to the raw user prompt."""
    if model_name.startswith("phi"):
        chat = [
            {"role": "system", "content": PHI_SYSTEM_MSG},
            {"role": "user", "content": raw_prompt},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif model_name.startswith("deepseek"):
        return raw_prompt
    elif model_name.startswith("qwen"):
        chat = [{"role": "user", "content": raw_prompt}]
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    raise ValueError(f"Model {model_name} not recognised.")


def _build_input(model_name, formatted_prompt, think_part, early_ans, percentage):
    """Build the full model input string at a given exit percentage."""
    if model_name.startswith("phi"):
        if percentage == 0:
            return f"{formatted_prompt}<think>{early_ans}"
        return f"{formatted_prompt}{think_part}\n\n{early_ans}"
    elif model_name.startswith("deepseek"):
        if percentage == 0:
            return f"{formatted_prompt}<think>\n{early_ans}"
        return f"{formatted_prompt}<think>\n{think_part}\n{early_ans}"
    elif model_name.startswith("qwen"):
        if percentage == 0:
            return f"{formatted_prompt}{early_ans}"
        return f"{formatted_prompt}{think_part}\n{early_ans}"
    raise ValueError(f"Model {model_name} not recognised.")


def _output_dist_metrics(output, model):
    """Compute per-layer entropy, logit margin, and perplexity from generation output.

    Uses hidden states at the last input position of the first new token step,
    projected through model.model.norm + model.lm_head (logit-lens approach).
    """
    # output.hidden_states[0]: tuple of [batch, seq, hidden] per layer at first new token
    hs = torch.stack(output.hidden_states[0]).squeeze(1)[:, -1, :]  # [n_layers, hidden]

    with torch.no_grad():
        logits = model.lm_head(model.model.norm(hs))  # [n_layers, vocab]
    probs = F.softmax(logits.float(), dim=-1)

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

    topk = probs.topk(2, dim=-1)
    logit_margin = topk.values[:, 0] - topk.values[:, 1]
    perplexity = 1.0 / (topk.values[:, 0] + 1e-12)

    del hs, logits, probs, topk
    return {
        "entropy": entropy.cpu().float().tolist(),
        "logit_margin": logit_margin.cpu().float().tolist(),
        "perplexity": perplexity.cpu().float().tolist(),
    }


def compute_early_exit(model_name, dataset, path, results_file=None, only_final=False, cache_dir=None, cuda_devices="0,1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Model '{model_name}' not recognised. Choose from: {list(MODEL_SOURCES)}")
    if dataset not in N_NEW_TOKENS:
        raise ValueError(f"Dataset '{dataset}' not recognised. Choose from: {list(N_NEW_TOKENS)}")

    path = Path(path)
    model_source = MODEL_SOURCES[model_name]
    n_new_tokens = N_NEW_TOKENS[dataset]

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_source, device_map="auto", torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    model.generation_config = GenerationConfig(
        max_new_tokens=n_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    model.eval()

    use_custom_json = results_file is not None
    if results_file is not None:
        with open(results_file, "r", encoding="utf-8") as f:
            data = pd.DataFrame(json.load(f))
    else:
        report_path = path / f"results/eureka_reports/{dataset}/{model_name}.jsonl"
        with open(report_path, "r", encoding="utf-8") as f:
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

    percentages = torch.tensor([1.0]) if only_final else torch.arange(0, 1.1, 0.1)
    early_ans = _get_suffix(model_name, dataset)

    results = defaultdict(list)
    for _, i_data in tqdm(enumerate(data.itertuples()), total=len(data)):
        if use_custom_json:
            think_ans_raw = i_data.think_answer
            if not isinstance(think_ans_raw, str):
                print(f"Skipping data point {i_data.data_point_id}: think_answer is None.")
                continue
            think_tokens = tokenizer.encode(think_ans_raw)
            ground_truth = str(i_data.ground_truth)
            formatted_prompt = _format_prompt(model_name, i_data.raw_prompt, tokenizer)
        else:
            raw_prompt = i_data.prompt
            try:
                ans = i_data.raw_model_output
            except AttributeError:
                ans = i_data.model_output

            if ans is None:
                print(f"Skipping data point {i_data.data_point_id}: model output is None.")
                continue

            think_tokens = tokenizer.encode(ans.split("</think>")[0])
            ground_truth = str(i_data.ground_truth)
            formatted_prompt = _format_prompt(model_name, raw_prompt, tokenizer)

        results["data_point_id"].append(i_data.data_point_id)
        results["data_repeat_id"].append(i_data.data_repeat_id)
        results["ground_truth"].append(ground_truth)

        for percentage in percentages:
            p = int(percentage * 100)
            p_str = str(p)

            if percentage == 0:
                think_part = ""
                results[f"think_n_tokens_{p_str}"].append(0)
            else:
                idx = int(len(think_tokens) * percentage)
                chunk = think_tokens[:idx]
                results[f"think_n_tokens_{p_str}"].append(len(chunk))
                think_part = tokenizer.decode(chunk)

            llm_input = _build_input(model_name, formatted_prompt, think_part, early_ans, percentage)
            input_ids = tokenizer(llm_input, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                output = model.generate(input_ids)

            output_text = tokenizer.decode(output.sequences[0][input_ids.shape[1]:])
            results[f"acc_{p_str}"].append(ground_truth in output_text)

            dist = _output_dist_metrics(output, model)
            for key, vals in dist.items():
                results[f"{key}_{p_str}"].append(vals)

            del output, input_ids
            torch.cuda.empty_cache()

    suffix = "_only_final" if only_final else ""
    output_file = path / f"results/early_exit/{dataset}/{model_name}_output_properties{suffix}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute output-distribution baselines at early-exit percentages.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_SOURCES))
    parser.add_argument("--dataset", type=str, required=True, choices=list(N_NEW_TOKENS),
                        help="Dataset name (determines answer suffix and token budget).")
    parser.add_argument("--results_file", type=str, default=None,
                        help=(
                            "Path to a pre-generated JSON results file produced by generate_tsp.py. "
                            "When omitted, loads from results/eureka_reports/<dataset>/<model>.jsonl."
                        ))
    parser.add_argument("--path", type=str, required=True, help="Path to the project root directory.")
    parser.add_argument("--only_final", action="store_true", help="Only compute at 100%% of thinking.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    args = parser.parse_args()

    compute_early_exit(
        model_name=args.model,
        dataset=args.dataset,
        path=args.path,
        results_file=args.results_file,
        only_final=args.only_final,
        cache_dir=args.cache_dir,
        cuda_devices=args.cuda_devices,
    )
