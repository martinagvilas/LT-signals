"""Generate model responses for BigBench tasks used in the paper (Appendix B).

Covers four tasks: codenames, fables (understanding_fables), com2sense, social_iqa.
Each task config defines how to load samples, build the user-side prompt content,
and score the model's final answer. The model-side prompt wrapping (chat template /
DeepSeek <think> prefix) is shared across tasks.

Usage:
    python generate_bigbench.py --model phi4-reasoning-plus --task fables \
        --benchmarks_path /path/to/BIG-bench --path /path/to/project \
        --cuda_devices 0,1
"""

import argparse
import json
import os
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_SOURCES = {
    "phi4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
    "deepseek-r1-qwen14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1-qwen32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1-llama8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1-llama70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "qwen3-14B": "Qwen/Qwen3-14B",
}

SAMPLING_PARAMS = {
    "phi4-reasoning-plus": SamplingParams(temperature=0.8, top_k=50, top_p=0.95, max_tokens=32768),
    "deepseek-r1-qwen14B": SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768),
    "deepseek-r1-qwen32B": SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768),
    "deepseek-r1-llama8B": SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768),
    "deepseek-r1-llama70B": SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768),
    "qwen3-14B": SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768),
}

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


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

def _load_task_json(benchmarks_path, relative):
    with open(Path(benchmarks_path) / relative) as f:
        return json.load(f)


def _parse_codenames(benchmarks_path):
    data = _load_task_json(benchmarks_path, "bigbench/benchmark_tasks/codenames/task.json")
    samples = []
    for idx, ex in enumerate(data["examples"]):
        target_words = [w.strip() for w in ex["target"].split(",")]
        samples.append({
            "data_point_id": idx,
            "user_content": f"{ex['input']} Provide your final answer in the format: 'Final Answer: [word1, word2, ...].'",
            "target": target_words,
        })
    return samples


def _score_codenames(final_ans, target):
    match = re.search(r'\[(.*?)\]', final_ans)
    words = match.group(1).lower() if match else final_ans.split("Final Answer")[-1].lower()
    return sum(1 for w in target if w in words) / len(target)


def _parse_fables(benchmarks_path):
    data = _load_task_json(benchmarks_path, "bigbench/benchmark_tasks/understanding_fables/task.json")
    task_prompt = (
        "Identify the most suitable moral for a given fable. "
        "Return only the number corresponding to your choice, in the format: Final Answer: [choice]."
    )
    samples = []
    for idx, ex in enumerate(data["examples"]):
        label = [i + 1 for i, v in enumerate(ex["target_scores"].values()) if v == 1][0]
        choices = [f"{i + 1}. {c}" for i, c in enumerate(ex["target_scores"].keys())]
        samples.append({
            "data_point_id": idx,
            "user_content": f"{task_prompt}\nFable: {ex['input']}\nChoices:\n" + "\n".join(choices),
            "target": label,
        })
    return samples


def _score_choice(final_ans, target):
    pred = re.search(r'Final Answer:\s*(\d+)', final_ans)
    return 1.0 if pred and str(target) in pred.group(1) else 0.0


def _parse_com2sense(benchmarks_path):
    data = _load_task_json(benchmarks_path, "bigbench/benchmark_tasks/com2sense/big-bench-data.json")
    samples = []
    for idx, ex in enumerate(data["examples"]):
        target = "yes" if ex["label"] == "True" else "no"
        samples.append({
            "data_point_id": idx,
            "user_content": f"Does the following sentence make commonsense? Answer strictly 'yes' or 'no'.\n\nSentence: {ex['sent']}",
            "target": target,
        })
    return samples


def _score_com2sense(final_ans, target):
    ans_lower = final_ans.lower()
    pred = "yes" if "yes" in ans_lower else ("no" if "no" in ans_lower else None)
    return 1.0 if pred == target else 0.0


def _parse_social_iqa(benchmarks_path):
    data = _load_task_json(benchmarks_path, "bigbench/benchmark_tasks/social_iqa/task.json")
    task_prompt = (
        "Answer a question about the motivations, emotional reactions, and preceding and following "
        "events surrounding the following interpersonal situation. "
        "Return only the index of the final answer in the format: Final Answer: [1/2/3...]."
    )
    samples = []
    for idx, ex in enumerate(data["examples"]):
        label = [i + 1 for i, v in enumerate(ex["target_scores"].values()) if v == 1][0]
        options = [f"{i + 1}. {o}" for i, o in enumerate(ex["target_scores"].keys())]
        samples.append({
            "data_point_id": idx,
            "user_content": f"{task_prompt}\nQuestion: {ex['input']}\nChoices:\n" + "\n".join(options),
            "target": label,
        })
    return samples


TASK_CONFIGS = {
    "codenames":  (_parse_codenames,  _score_codenames, "codenames"),
    "fables":     (_parse_fables,     _score_choice,    "fables"),
    "com2sense":  (_parse_com2sense,  _score_com2sense, "com2sense"),
    "social_iqa": (_parse_social_iqa, _score_choice,    "social-iqa"),
}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(model_name, user_content, tokenizer):
    if model_name.startswith("phi"):
        chat = [
            {"role": "system", "content": PHI_SYSTEM_MSG},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    elif model_name.startswith("deepseek"):
        return f"{user_content}<think>\n"
    elif model_name.startswith("qwen"):
        chat = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    else:
        raise ValueError(f"Model {model_name} not recognised.")


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_bigbench_answers(model_name, task, benchmarks_path, path, cuda_devices="0,1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Model '{model_name}' not recognised. Choose from: {list(MODEL_SOURCES)}")
    if task not in TASK_CONFIGS:
        raise ValueError(f"Task '{task}' not recognised. Choose from: {list(TASK_CONFIGS)}")

    parse_fn, score_fn, output_dir = TASK_CONFIGS[task]

    model_source = MODEL_SOURCES[model_name]
    sampling_params = SAMPLING_PARAMS[model_name]

    download_dir = Path(path) / "llms"
    download_dir.mkdir(parents=True, exist_ok=True)

    num_devices = len(cuda_devices.split(","))
    llm = LLM(model=model_source, tensor_parallel_size=num_devices, dtype="bfloat16", download_dir=str(download_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    samples = parse_fn(benchmarks_path)

    results = []
    for sample in tqdm(samples):
        prompt = build_prompt(model_name, sample["user_content"], tokenizer)
        outputs = llm.generate([prompt] * 5, sampling_params, use_tqdm=False)

        for o_idx, o in enumerate(outputs):
            try:
                ans = o.outputs[0].text
                think_ans, final_ans = ans.split("</think>", 1)
                accuracy = score_fn(final_ans, sample["target"])
                results.append({
                    "prompt": prompt,
                    "data_point_id": sample["data_point_id"],
                    "data_repeat_id": o_idx,
                    "target": sample["target"],
                    "think_answer": think_ans,
                    "final_answer": final_ans,
                    "accuracy": accuracy,
                })
            except Exception:
                results.append({
                    "prompt": prompt,
                    "data_point_id": sample["data_point_id"],
                    "data_repeat_id": o_idx,
                    "target": sample["target"],
                    "think_answer": None,
                    "final_answer": None,
                    "accuracy": 0.0,
                })

    output_path = Path(path) / "results" / "bigbench" / output_dir / f"{model_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BigBench responses for LT-signals paper.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_SOURCES))
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_CONFIGS))
    parser.add_argument("--benchmarks_path", type=str, required=True, help="Path to the BIG-bench repo root.")
    parser.add_argument("--path", type=str, required=True, help="Path to the project root directory.")
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    args = parser.parse_args()

    generate_bigbench_answers(args.model, args.task, args.benchmarks_path, args.path, args.cuda_devices)
