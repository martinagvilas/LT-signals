"""Generate model responses for the TSP task (Appendix B).

TSP results for phi4-reasoning-plus come from Eureka reports directly. This
script generates responses for the remaining models (deepseek-r1-llama8B,
deepseek-r1-qwen32B) by re-using the prompts from phi4's Eureka report, since
those models do not have their own Eureka TSP reports.

Usage:
    python generate_tsp.py --model deepseek-r1-llama8B \
        --path /path/to/project --cuda_devices 0,1
"""

import argparse
import json
import os
import pandas as pd
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


def _score_tsp(final_ans, target):
    pred = final_ans.split("TotalDistance")[-1].strip()
    return 1.0 if str(target) in pred else 0.0


def generate_tsp_answers(model_name, path, cuda_devices="0,1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Model '{model_name}' not recognised. Choose from: {list(MODEL_SOURCES)}")

    path = Path(path)
    model_source = MODEL_SOURCES[model_name]
    sampling_params = SAMPLING_PARAMS[model_name]

    download_dir = path / "llms"
    download_dir.mkdir(parents=True, exist_ok=True)

    num_devices = len(cuda_devices.split(","))
    llm = LLM(model=model_source, tensor_parallel_size=num_devices, dtype="bfloat16", download_dir=str(download_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    # Load prompts from phi4's Eureka report (one prompt per data point)
    phi4_report = path / "results/eureka_reports/TSP/phi4-reasoning-plus.jsonl"
    with open(phi4_report, "r", encoding="utf-8") as f:
        data = pd.DataFrame([json.loads(line) for line in f])

    data = data.loc[data["data_repeat_id"].isin([f"repeat_{i}" for i in range(5)])]
    ids = (
        data.drop_duplicates("data_point_id")[["data_point_id", "category"]]
        .groupby("category", group_keys=False)
        .apply(lambda g: g.sample(n=20, random_state=0))["data_point_id"]
    )
    data = data[data["data_point_id"].isin(ids)]
    data = data.loc[data["data_repeat_id"] == "repeat_0"]

    results = []
    for sample_idx, sample in tqdm(enumerate(data.itertuples()), total=len(data)):
        prompt = build_prompt(model_name, sample.prompt, tokenizer)
        target = sample.ground_truth
        outputs = llm.generate([prompt] * 5, sampling_params, use_tqdm=False)

        for o_idx, o in enumerate(outputs):
            try:
                ans = o.outputs[0].text
                think_ans, final_ans = ans.split("</think>", 1)
                accuracy = _score_tsp(final_ans, target)
                results.append({
                    "prompt": prompt,
                    "raw_prompt": sample.prompt,
                    "data_point_id": sample.data_point_id,
                    "data_repeat_id": f"repeat_{o_idx}",
                    "ground_truth": target,
                    "think_answer": think_ans,
                    "final_answer": final_ans,
                    "accuracy": accuracy,
                    "think_length": len(tokenizer.encode(think_ans)),
                })
            except Exception:
                results.append({
                    "prompt": prompt,
                    "raw_prompt": sample.prompt,
                    "data_point_id": sample.data_point_id,
                    "data_repeat_id": f"repeat_{o_idx}",
                    "ground_truth": target,
                    "think_answer": None,
                    "final_answer": None,
                    "accuracy": 0.0,
                    "think_length": 0,
                })

        if (sample_idx + 1) % 5 == 0:
            checkpoint_path = path / "results" / "TSP" / f"{model_name}_results_partial.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(results, f, indent=4)

    output_path = path / "results" / "TSP" / f"{model_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSP responses for LT-signals paper.")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_SOURCES))
    parser.add_argument("--path", type=str, required=True, help="Path to the project root directory.")
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    args = parser.parse_args()

    generate_tsp_answers(args.model, args.path, args.cuda_devices)
