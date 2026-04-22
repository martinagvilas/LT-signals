"""Compute early-exit accuracy and last-layer logit confidence at fixed token budgets.

For each sample, truncates the thinking trace at a fixed number of tokens
(2000, 3000, ..., 10000) and forces the model to answer. Records:

  - ntoken:        token budget used
  - acc:           accuracy (ground truth in output)
  - top1_prob:     top-1 softmax probability of the last layer
  - logit_margin:  top-2 logit margin of the last layer

Output: results/early_exit/{dataset}/{model_name}_ntokens_exit.json

Usage:
    python compute_early_exit_ntokens.py --model phi4-reasoning-plus \\
        --dataset GPQA --path /path/to/project --cuda_devices 0,1
"""

import argparse
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

EARLY_EXIT_SUFFIX = {
    "phi": {
        "GPQA":     "I'll produce final answer</think>\n Final answer:",
        "AIME2025": "I'll produce final answer</think>\nFinal answer: ",
        "TSP":      "I'll produce final answer</think>\n'Total distance': ",
    },
    "deepseek": {
        "GPQA":     "\n\n</think>\nFinal answer: \\boxed{",
        "AIME2025": "\n\n</think>\nFinal answer: \\boxed{",
        "TSP":      "\n\n</think>\n'Total distance': ",
    },
    "qwen": {
        "GPQA":     "\n</think>\n\nFinal answer:",
        "AIME2025": "\n</think>\n\nFinal answer:",
        "TSP":      "\n\n</think>\n'TotalDistance': ",
    },
}


def _get_suffix(model_name, dataset):
    for prefix, suffixes in EARLY_EXIT_SUFFIX.items():
        if model_name.startswith(prefix):
            return suffixes[dataset]
    raise ValueError(f"No early-exit suffix for model={model_name}, dataset={dataset}")


def _format_prompt(model_name, raw_prompt, tokenizer):
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


def _build_input(model_name, formatted_prompt, think_part, early_ans):
    if model_name.startswith("phi"):
        return f"{formatted_prompt}{think_part}\n\n{early_ans}"
    elif model_name.startswith("deepseek"):
        return f"{formatted_prompt}<think>\n{think_part}\n{early_ans}"
    elif model_name.startswith("qwen"):
        return f"{formatted_prompt}{think_part}\n{early_ans}"
    raise ValueError(f"Model {model_name} not recognised.")


def compute_early_exit_ntokens(model_name, dataset, path, cache_dir=None, cuda_devices="0,1"):
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

    ntokens = torch.arange(2000, 11000, 1000)
    early_ans = _get_suffix(model_name, dataset)

    results = []
    for _, i_data in tqdm(enumerate(data.itertuples()), total=len(data)):
        raw_prompt = i_data.prompt
        try:
            ans = i_data.raw_model_output
        except AttributeError:
            ans = i_data.model_output

        if ans is None:
            print(f"Skipping data point {i_data.data_point_id}: model output is None.")
            continue

        think_tokens = tokenizer.encode(ans.split("</think>")[0])
        if len(think_tokens) < int(ntokens[0]):
            print(f"Skipping {i_data.data_point_id}: thinking trace too short ({len(think_tokens)} tokens).")
            continue

        ground_truth = str(i_data.ground_truth)
        formatted_prompt = _format_prompt(model_name, raw_prompt, tokenizer)

        for ntoken in ntokens:
            if ntoken >= len(think_tokens):
                break

            think_part = tokenizer.decode(think_tokens[:ntoken])
            llm_input = _build_input(model_name, formatted_prompt, think_part, early_ans)
            input_ids = tokenizer(llm_input, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                output = model.generate(input_ids)

            output_text = tokenizer.decode(output.sequences[0][input_ids.shape[1]:])
            acc = ground_truth in output_text

            # Last-layer logit confidence
            hs = torch.stack(output.hidden_states[0]).squeeze(1)[:, -1, :]  # [n_layers, hidden]
            with torch.no_grad():
                logits = model.lm_head(model.model.norm(hs))
            probs = F.softmax(logits[-1].float(), dim=-1)  # last layer only
            topk = probs.topk(2)

            results.append({
                "data_point_id": i_data.data_point_id,
                "data_repeat_id": i_data.data_repeat_id,
                "ntoken": int(ntoken),
                "ground_truth": ground_truth,
                "output": output_text,
                "acc": acc,
                "top1_prob": topk.values[0].item(),
                "logit_margin": (topk.values[0] - topk.values[1]).item(),
            })

            del output, hs, logits, probs, topk, input_ids
            torch.cuda.empty_cache()

    output_file = path / f"results/early_exit/{dataset}/{model_name}_ntokens_exit.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute early-exit accuracy at fixed token budgets.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_SOURCES))
    parser.add_argument("--dataset", type=str, required=True, choices=list(N_NEW_TOKENS))
    parser.add_argument("--path", type=str, required=True, help="Path to the project root directory.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    args = parser.parse_args()

    compute_early_exit_ntokens(
        model_name=args.model,
        dataset=args.dataset,
        path=args.path,
        cache_dir=args.cache_dir,
        cuda_devices=args.cuda_devices,
    )
