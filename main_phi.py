import argparse
import json
import os
import time
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from tqdm import tqdm

# Import local config
try:
    from config import *
except ImportError:
    # Fallback defaults if config.py is missing
    DATASET_PATH = "data/nq_dev_augmented_translated.jsonl"
    MAX_EXAMPLES = None
    GEN_TEMPERATURE = 0.2
    GEN_MAX_TOKENS = 256
    JUDGE_MODEL = "gpt-5-mini"  # Adjusted to common model naming
    JUDGE_TEMPERATURE = 0

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

PHI4_MODEL_ID = "microsoft/phi-4"


def load_local_generator(model_name: str):
    """
    Phi-4 only loader, tuned to run reliably on a 4090 (24GB) using 4-bit NF4 quantization.
    Methodology mirrors the previously-working setup:
      - 4-bit (NF4) + double quant
      - compute dtype float16 (slightly lower memory than bf16)
      - keep everything on GPU (cuda:0)
      - low_cpu_mem_usage=True
    """
    if model_name != PHI4_MODEL_ID:
        raise ValueError(
            f"This script is configured for Phi-4 only. "
            f"Pass model='{PHI4_MODEL_ID}' (got '{model_name}')."
        )

    print(f"--- Loading Model (Phi-4): {model_name} ---")

    # Optional HF auth
    auth = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        **auth,
    )

    # Keep padding behavior consistent with your existing pipeline setup
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **auth,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def generate_answers_batch(questions_lb: list[str], gen_pipe) -> list[str]:
    """
    Uses apply_chat_template to ensure Phi-4 receives the correct special tokens.

    IMPORTANT: Output structure + methodology kept the same:
      - formatted_prompts list
      - pipeline(...) call
      - max_new_tokens=GEN_MAX_TOKENS
      - temperature=GEN_TEMPERATURE
      - do_sample=GEN_TEMPERATURE > 0
      - return_full_text=False
      - pad_token_id=eos
    """
    formatted_prompts = []
    for q in questions_lb:
        messages = [
            {
                "role": "system",
                "content": "You are a Luxembourgish voice assistant. Respond in Luxembourgish. Keep replies concise (but in a full sentence) and relevant to the user's query. Do not add disclaimers."
            },
            {"role": "user", "content": q},
        ]

        prompt = gen_pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(prompt)

    outputs = gen_pipe(
        formatted_prompts,
        max_new_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        do_sample=GEN_TEMPERATURE > 0,
        return_full_text=False,
        pad_token_id=gen_pipe.tokenizer.eos_token_id,
    )

    return [out[0].get("generated_text", "").strip() for out in outputs]


# OpenAI client for judging
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """
You are an expert evaluator of a Luxembourgish voice assistant.
Use the provided 'References' as the ground truth for factual correctness. 

Score the answer on a scale from 0 to 5 for EACH criterion.

1) Factual correctness
2) Completeness
3) No hallucination
4) Clarity & helpfulness
5) Luxembourgish language quality

Return ONLY valid JSON with these keys: 
factual_correctness, completeness, hallucination, clarity, lux_quality, overall, fatal_error
"""


def judge_answer(question, model_answer, reference_answers):
    """Individual judging for maximum reliability."""
    try:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user",
                 "content": f"Question: {question}\nModel Answer: {model_answer}\nReferences: {reference_answers}"}
            ],
            response_format={"type": "json_object"},
            temperature=JUDGE_TEMPERATURE,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error during judging: {e}")
        return {"error": str(e)}


def load_dataset(path, limit=None):
    data = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Must be microsoft/phi-4 (Phi-4 only)")
    args = parser.parse_args()

    model_name = args.model

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gen_pipe = load_local_generator(model_name)
    data = load_dataset(DATASET_PATH, MAX_EXAMPLES)

    # Use a small batch size for generation stability
    BATCH_SIZE = 2
    results = []

    # Metrics
    start_time = time.time()

    safe_model = model_name.replace("/", "__")
    out_path = f"results_{safe_model}.jsonl"

    print(f"Starting evaluation on {len(data)} examples...")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Evaluating"):
            batch = data[i:i + BATCH_SIZE]
            questions = [ex["question_lb"] for ex in batch]

            # 1. Generate
            try:
                answers = generate_answers_batch(questions, gen_pipe)
            except Exception as e:
                print(f"Generation error: {e}")
                answers = ["ERROR_GENERATING"] * len(questions)

            # 2. Judge and Save incrementally
            for ex, answer in zip(batch, answers):
                reference = ex.get("answers_lb", [])
                scores = judge_answer(ex["question_lb"], answer, reference, )

                result = {
                    "id": ex.get("id"),
                    "question": ex["question_lb"],
                    "reference": reference,
                    "model": model_name,
                    "answer": answer,
                    "scores": scores,
                    "timestamp": time.time()
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()  # Ensure it's written if the script crashes

    total_time = time.time() - start_time
    print(f"\nDone! Results saved to {out_path}")
    print(f"Total time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
