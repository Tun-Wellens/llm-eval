import argparse
import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from google import genai
from google.genai import types

# Import local config
try:
    from config import *
except ImportError:
    DATASET_PATH = "data/nq_dev_augmented_translated.jsonl"
    MAX_EXAMPLES = None
    GEN_TEMPERATURE = 0.2
    GEN_MAX_TOKENS = 1024 
    JUDGE_MODEL = "gpt-5-mini"
    JUDGE_TEMPERATURE = 0

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=GOOGLE_API_KEY)

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

def generate_single_answer(question: str, model_name: str) -> str:
    """
    Generates a single answer with RETRY LOGIC for 503/Overloaded errors.
    """
    
    # Define Config
    generate_config = types.GenerateContentConfig(
        temperature=GEN_TEMPERATURE,
        max_output_tokens=1024, # Force high limit
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        system_instruction="You are a Luxembourgish voice assistant. Respond in Luxembourgish. Keep replies concise (but in a full sentence) and relevant to the user's query. Do not add disclaimers.",
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),
        ]
    )

    max_retries = 8 # High retry count for preview models
    base_wait_time = 2 

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=question,
                config=generate_config
            )
            
            if response.text:
                return response.text.strip()
            
            # Handle empty text (Safety/Recitation) - Do NOT retry these
            reason = "UNKNOWN"
            if response.candidates:
                reason = response.candidates[0].finish_reason
            
            print(f"\n[BLOCKED] Query '{question}' -> Finish Reason: {reason}")
            return f"BLOCKED_{reason}"

        except Exception as e:
            error_str = str(e)
            # Retry on 503 (Overloaded) or 429 (Rate Limit)
            if "503" in error_str or "429" in error_str or "Overloaded" in error_str:
                wait_time = base_wait_time * (2 ** attempt) # 2, 4, 8, 16...
                # Cap wait time at 60 seconds
                wait_time = min(wait_time, 60)
                print(f"\n[WAIT] Server overloaded. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                # Fatal error (e.g. Auth failure, Bad Request)
                print(f"\n[ERROR] Query '{question}' -> Fatal API Error: {e}")
                return "ERROR_GENERATING"

    print(f"\n[GAVE UP] Query '{question}' failed after {max_retries} retries.")
    return "ERROR_TIMEOUT"

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
    parser.add_argument("model", nargs="?", default="gemini-3-flash-preview", help="Gemini model name")
    args = parser.parse_args()
    
    model_name = args.model
    data = load_dataset(DATASET_PATH, MAX_EXAMPLES)
    start_time = time.time()

    safe_model = model_name.replace("/", "__").replace("-", "_")
    out_path = f"results_{safe_model}.jsonl"

    print(f"Starting evaluation on {len(data)} examples using {model_name}...")
    print(f"Mode: Sequential with Auto-Retry | Max Tokens: 1024")

    # Open file once
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, ex in enumerate(tqdm(data, desc="Evaluating")):
            
            question = ex["question_lb"]
            reference = ex.get("answers_lb", [])

            # 1. Generate (with retries)
            answer = generate_single_answer(question, model_name)

            # 2. Judge
            scores = judge_answer(question, answer, reference)
            
            # 3. Save
            result = {
                "id": ex.get("id"),
                "question": question,
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