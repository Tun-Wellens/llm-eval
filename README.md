# LLM Evaluation 

## Structure

**Main Scripts:**
- `main.py`, `main_api.py`, `main_ministral.py`, `main_phi.py` - Run evaluations on different models

**Annotation & Analysis:**
- `human_annotator.py` - Streamlit app for manual evaluation
```bash
uv run streamlit run human_annotator.py
```
- `evaluation.ipynb` - Comprehensive analysis of results and judge performance

**Data:**
- `data/` - Input datasets (NQ-open questions, augmented versions, translations)
- `results/` - Evaluation results (JSONL files per model, human annotations)

**Config:**
- `config.py` - Configuration settings
- `pyproject.toml` - Dependencies

**Other:**
- `job_logs/` - HPC job logs (SLURM)

**dotenv**
```
OPENAI_API_KEY = sk-proj-Aaaabbbcccc

GOOGLE_API_KEY = Aaaabbbcccc

HUGGINGFACEHUB_API_TOKEN = hf_Aaaabbbcccc