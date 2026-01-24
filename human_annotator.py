"""
Streamlit application for human annotation of LLM evaluation results.
Implements blind evaluation to prevent bias from model names.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import random

RESULTS_DIR = Path("results")
ANNOTATIONS_FILE = Path("human_annotations.jsonl")
SAMPLES_PER_MODEL = 20
NUM_MODELS = 5
TARGET_TOTAL = SAMPLES_PER_MODEL * NUM_MODELS

CRITERIA = [
    "factual_correctness",
    "completeness",
    "no_hallucination",
    "clarity_helpfulness",
    "lux_language_quality",
]

CRITERIA_LABELS = {
    "factual_correctness": "Factual Correctness",
    "completeness": "Completeness",
    "no_hallucination": "No Hallucination",
    "clarity_helpfulness": "Clarity & Helpfulness",
    "lux_language_quality": "Luxembourgish Language Quality",
}


def load_all_results() -> Dict[str, List[Dict]]:
    """Load all result files from the results directory."""
    all_results = {}
    
    for result_file in RESULTS_DIR.glob("results_*.jsonl"):
        model_name = result_file.stem.replace("results_", "")
        results = []
        
        with open(result_file, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if results:
            all_results[model_name] = results
    
    return all_results


def load_existing_annotations() -> set:
    """Load already annotated sample IDs to avoid duplicates."""
    annotated_ids = set()
    
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Create a unique key: (model_name, sample_id)
                    key = (record.get("model"), record.get("id"))
                    annotated_ids.add(key)
                except json.JSONDecodeError:
                    continue
    
    return annotated_ids


def get_stratified_samples(all_results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Get stratified samples: 20 samples from each of the 5 models.
    Exclude already-annotated samples.
    """
    annotated_ids = load_existing_annotations()
    samples = []
    
    for model_name, results in all_results.items():
        # Filter out already-annotated samples
        available = [
            r for r in results 
            if (model_name, r["id"]) not in annotated_ids
        ]
        
        # Randomly sample up to SAMPLES_PER_MODEL
        num_to_sample = min(SAMPLES_PER_MODEL, len(available))
        if num_to_sample > 0:
            sampled = random.sample(available, num_to_sample)
            samples.extend(sampled)
    
    # Shuffle to avoid model-specific ordering bias
    random.shuffle(samples)
    
    return samples


def count_completed_annotations() -> int:
    """Count how many annotations have been completed."""
    if not ANNOTATIONS_FILE.exists():
        return 0
    
    count = 0
    with open(ANNOTATIONS_FILE, "r") as f:
        for line in f:
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue
    
    return count


def save_annotation(annotation: Dict) -> None:
    """Append annotation to the JSONL file (atomic write)."""
    with open(ANNOTATIONS_FILE, "a") as f:
        f.write(json.dumps(annotation) + "\n")



def initialize_session_state():
    """Initialize Streamlit session state."""
    if "samples" not in st.session_state:
        st.session_state.samples = []
        st.session_state.current_idx = 0
        st.session_state.slider_values = {criterion: 3 for criterion in CRITERIA}
        st.session_state.load_samples()
    
    if "completed" not in st.session_state:
        st.session_state.completed = count_completed_annotations()


def load_samples():
    """Load stratified samples into session state."""
    all_results = load_all_results()
    st.session_state.samples = get_stratified_samples(all_results)
    st.session_state.current_idx = 0


st.session_state.load_samples = load_samples


def main():
    st.set_page_config(
        page_title="LLM Evaluation - Human Annotator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("LLM Evaluation - Human Annotation Tool")
    st.markdown(
        "Please evaluate the following LLM responses. "
        "The model name is intentionally hidden to prevent bias."
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Progress section
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        progress = st.session_state.completed / TARGET_TOTAL
        st.progress(min(progress, 1.0), text=f"{st.session_state.completed}/{TARGET_TOTAL}")
    
    with col2:
        st.metric("Completed", st.session_state.completed, delta=TARGET_TOTAL - st.session_state.completed)
    
    with col3:
        if st.session_state.completed >= TARGET_TOTAL:
            st.success("All 100 samples annotated!")
            st.balloons()
            return
    
    st.divider()
    
    # Check if we have samples available
    if not st.session_state.samples:
        st.warning("No more samples available to annotate!")
        if st.button("Refresh and reload samples"):
            st.session_state.load_samples()
            st.rerun()
        return
    
    # Get current sample
    current_sample = st.session_state.samples[st.session_state.current_idx]
    
    # Display sample info
    st.markdown("### Sample Information")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"**Sample ID:** {current_sample['id']}")
        st.markdown(f"**Progress:** {st.session_state.current_idx + 1}/{len(st.session_state.samples)}")
    
    with col2:
        st.markdown(f"**Question/Prompt:**\n> {current_sample['question']}")
    
    # Display reference/ground truth
    if current_sample.get("reference"):
        st.markdown("### Reference Answer (Ground Truth)")
        reference = current_sample["reference"]
        if isinstance(reference, list):
            st.markdown("Acceptable answers: " + ", ".join([f"**{r}**" for r in reference]))
        else:
            st.markdown(f"**{reference}**")
    
    st.divider()
    
    # Display response (answer field)
    st.markdown("### LLM Response")
    st.text_area(
        label="Response",
        value=current_sample.get("answer", "N/A"),
        height=150,
        disabled=True,
        label_visibility="collapsed",
    )
    
    st.divider()
    
    # Evaluation sliders
    st.markdown("### Evaluation Criteria")
    st.markdown("Please rate the response on the following dimensions (0-5 scale):")
    
    # Create sliders for each criterion
    for criterion in CRITERIA:
        st.session_state.slider_values[criterion] = st.slider(
            label=CRITERIA_LABELS[criterion],
            min_value=0,
            max_value=5,
            value=st.session_state.slider_values[criterion],
            step=1,
            key=f"slider_{criterion}",
        )
    
    st.divider()
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Submit Annotation", use_container_width=True):
            # Prepare annotation record
            annotation = {
                "annotated_at": datetime.now().isoformat(),
                "sample_id": current_sample["id"],
                "model": current_sample["model"],
                "question": current_sample["question"],
                "answer": current_sample["answer"],
                "annotations": {criterion: st.session_state.slider_values[criterion] for criterion in CRITERIA},
                "human_score": sum(st.session_state.slider_values[criterion] for criterion in CRITERIA) / len(CRITERIA),
            }
            
            # Save to file
            save_annotation(annotation)
            
            # Update counters
            st.session_state.completed += 1
            
            # Move to next sample or finish
            if st.session_state.current_idx + 1 < len(st.session_state.samples):
                st.session_state.current_idx += 1
                # Reset sliders for next sample
                st.session_state.slider_values = {criterion: 3 for criterion in CRITERIA}
                st.success("Annotation saved! Loading next sample...")
                st.rerun()
            else:
                st.success("Annotation saved!")
                st.info("No more samples in this batch. Reloading...")
                st.session_state.load_samples()
                st.rerun()
    
    with col2:
        if st.button("Skip", use_container_width=True):
            if st.session_state.current_idx + 1 < len(st.session_state.samples):
                st.session_state.current_idx += 1
                st.session_state.slider_values = {criterion: 3 for criterion in CRITERIA}
                st.rerun()
            else:
                st.info("No more samples in this batch.")
    
    with col3:
        st.markdown("")


if __name__ == "__main__":
    main()
