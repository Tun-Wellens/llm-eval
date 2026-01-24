import argparse
import json
import os
import time
import torch
import requests
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from transformers import (
    AutoModelForImageTextToText, 
    AutoProcessor, 
    AutoTokenizer, 
    FineGrainedFP8Config
)
from tqdm import tqdm

# Import local config
try:
    from config import *
except ImportError:
    DATASET_PATH = "data/nq_dev_augmented_translated.jsonl"
    MAX_EXAMPLES = None
    GEN_TEMPERATURE = 0.2
    GEN_MAX_TOKENS = 256
    JUDGE_MODEL = "gpt-5-mini"
    JUDGE_TEMPERATURE = 0

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

class MinistralGenerator:
    def __init__(self, model_id, auth):
        print(f"--- Loading Ministral Model: {model_id} ---")
        print(f"--- Using {torch.cuda.device_count()} GPUs ---")
        
        self.processor = AutoProcessor.from_pretrained(model_id, **auth)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **auth)
        
        # MUST use dequantize=True for V100 compatibility
        quantization_config = FineGrainedFP8Config(dequantize=True) 

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16, # Use float16 for V100 (SXM2)
            quantization_config=quantization_config,
            trust_remote_code=True,
            **auth
        )
        self.model.eval()

    def __call__(self, prompts, **kwargs):
        results = []
        for prompt_text in prompts:
            inputs = self.processor(text=prompt_text, return_tensors="pt").to("cuda:0")
            
            # Match input dtype to model dtype (float16)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    inputs[k] = v.to(torch.float16)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 256),
                    temperature=kwargs.get("temperature", 0.2),
                    do_sample=kwargs.get("do_sample", True),
                    pad_token_id=self.tokenizer.eos_token_id
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            results.append([{"generated_text": generated_text}])
            
        return results

def load_local_generator(model_name: str):
    auth = {"token": hf_token} if hf_token else {}
    return MinistralGenerator(model_name, auth)

def generate_answers_batch(questions_lb: list[str], gen_pipe) -> list[str]:
    formatted_prompts = []
    for q in questions_lb:
        messages = [
            {
                "role": "system", 
                "content": "You are a Luxembourgish voice assistant. Respond in Luxembourgish. Keep replies concise (but in a full sentence) and relevant to the user's query. Do not add disclaimers."
            },
            {"role": "user", "content": q},
        ]
        
        prompt = gen_pipe.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )
        formatted_prompts.append(prompt)
    
    outputs = gen_pipe(
        formatted_prompts,
        max_new_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        do_sample=GEN_TEMPERATURE > 0,
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
    try:
        response = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"Question: {question}\nModel Answer: {model_answer}\nReferences: {reference_answers}"}
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
    parser.add_argument("model", help="HF repo or local path")
    args = parser.parse_args()
    
    model_name = args.model
    gen_pipe = load_local_generator(model_name)
    data = load_dataset(DATASET_PATH, MAX_EXAMPLES)

    BATCH_SIZE = 2 
    start_time = time.time()
    
    safe_model = model_name.replace("/", "__")
    out_path = f"results_{safe_model}.jsonl"

    print(f"Starting evaluation on {len(data)} examples...")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Evaluating"):
            batch = data[i:i + BATCH_SIZE]
            questions = [ex["question_lb"] for ex in batch]
            
            try:
                answers = generate_answers_batch(questions, gen_pipe)
            except Exception as e:
                print(f"Generation error: {e}")
                answers = ["ERROR_GENERATING"] * len(questions)

            for ex, answer in zip(batch, answers):
                reference = ex.get("answers_lb", [])
                scores = judge_answer(ex["question_lb"], answer, reference)
                
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
                fout.flush()

    total_time = time.time() - start_time
    print(f"\nDone! Results saved to {out_path}")
    print(f"Total time: {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()