import itertools
import streamlit as st
import pandas as pd
import concurrent.futures
import time
import json
import os
from typing import List, Dict, Any
from openai import RateLimitError, InternalServerError
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.ingestion import load_document, split_documents
from src.utils.vectorstore import create_vectorstore
from src.utils.llm_manager import get_llm, ensure_ollama_reachable, unload_ollama_model
from src.utils.rag_chain import get_rag_chain
from src.utils.judge import get_judge_chain

# Load Model Config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/model_config.json")
MODEL_CONFIG = {}
try:
    with open(CONFIG_PATH, "r") as f:
        MODEL_CONFIG = json.load(f)
except FileNotFoundError:
    print(f"Warning: Config file not found at {CONFIG_PATH}. Using defaults (no retry).")

def get_retry_config(model_name: str) -> Dict[str, Any]:
    """Gets retry config for a model, falling back to default."""
    for provider in ["Ollama", "Groq", "GitHub", "Gemini"]:
        if provider in model_name:
            return MODEL_CONFIG.get(provider, MODEL_CONFIG.get("default", {"enable_retry": False}))
    return MODEL_CONFIG.get("default", {"enable_retry": False})

def create_retryer(config: Dict[str, Any]):
    """Creates a tenacity Retrying object from config, or None if disabled."""
    if not config.get("enable_retry", False):
        return None
    return tenacity.Retrying(
        retry=retry_if_exception_type((RateLimitError, InternalServerError)),
        wait=wait_exponential(multiplier=2, min=config.get("wait_min", 4), max=config.get("wait_max", 60)),
        stop=stop_after_attempt(config.get("max_attempts", 5)),
        reraise=True
    )

def _run_generation(
    rag_chain, question: str, model_name: str
) -> Dict[str, Any]:
    """Runs the generation phase only. Returns answer and context."""
    try:
        gen_retry_config = get_retry_config(model_name)
        gen_retryer = create_retryer(gen_retry_config)
        
        start_rag = time.time()
        response = None
        if gen_retryer:
            for attempt in gen_retryer:
                with attempt:
                    response = rag_chain.invoke({"input": question})
        else:
            response = rag_chain.invoke({"input": question})
                    
        rag_time = time.time() - start_rag
        
        answer = response["answer"]
        context_text = "\n\n".join([doc.page_content for doc in response["context"]])
        
        return {
            "successful": True,
            "answer": answer,
            "context": context_text,
            "latency_rag": rag_time
        }
    except Exception as e:
        return {"successful": False, "error": str(e)}


def _run_judging(
    judge_chain, question: str, answer: str, context: str, judge_model: str
) -> Dict[str, Any]:
    """Runs the judging phase only. Returns scores."""
    try:
        eval_input = {"question": question, "answer": answer, "context": context}
        
        judge_retry_config = get_retry_config(judge_model)
        judge_retryer = create_retryer(judge_retry_config)
        
        start_judge = time.time()
        score = None
        if judge_retryer:
            for attempt in judge_retryer:
                with attempt:
                    score = judge_chain.invoke(eval_input)
        else:
            score = judge_chain.invoke(eval_input)
                    
        judge_time = time.time() - start_judge
        
        return {
            "successful": True,
            "score": score,
            "latency_judge": judge_time
        }
    except Exception as e:
        return {"successful": False, "error": str(e)}


def run_batch_experiment(
    file_paths: List[str],
    config: Dict[str, Any],
    question: str,
    progress_bar: Any = None,
    status_placeholder: Any = None
) -> List[Dict[str, Any]]:
    """
    Runs a batch of experiments based on the configuration grid.
    """
    
    # Check Ollama Health if needed
    if "Ollama" in config["model_name"] or "Ollama" in config["judge_model"]:
        if not ensure_ollama_reachable():
            st.error("Could not reach or start Ollama service. Please make sure Ollama is installed and running.")
            return []
    
    results = []
    
    # 1. Define Grid
    if file_paths:
        ingestion_params = list(itertools.product(config["chunk_sizes"], config["chunk_overlaps"]))
        retrieval_params = config["k_retrievals"]
    else:
        # No files = No RAG. Run once per model config.
        ingestion_params = [(None, None)]
        retrieval_params = [0]
        
    generation_params = list(itertools.product(config["temperatures"], config["top_ps"]))
    
    # Calculate total steps for progress bar
    total_steps = len(ingestion_params) * len(retrieval_params) * len(generation_params)
    current_step = 0
    
    # Pre-load all documents
    raw_docs = []
    if file_paths:
        for f_path in file_paths:
            try:
                raw_docs.extend(load_document(f_path))
            except Exception as e:
                 st.error(f"Failed to load file {f_path}: {e}")
                 return []
        
        if not raw_docs:
            st.error("No documents loaded.")
            return []

    # Get concurrency limit
    max_workers = config.get("max_concurrency", 1)

    for chunk_size, chunk_overlap in ingestion_params:
        
        # --- Ingestion Phase (Per Chunk Config) ---
        vectorstore = None
        
        if file_paths:
            if chunk_overlap >= chunk_size:
                print(f"Skipping invalid config: Size={chunk_size}, Overlap={chunk_overlap}")
                current_step += len(retrieval_params) * len(generation_params)
                if progress_bar:
                    try:
                        progress_bar.progress(min(current_step / total_steps, 1.0))
                    except: pass
                continue
            
            try:
                # Split
                chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                # Create VectorStore
                vectorstore = create_vectorstore(chunks)
                
            except Exception as e:
                st.error(f"Error during ingestion (Size={chunk_size}, Overlap={chunk_overlap}): {e}")
                current_step += len(retrieval_params) * len(generation_params)
                continue
            
        # --- Build task list ---
        tasks = []
        for k in retrieval_params:
            for temperature, top_p in generation_params:
                tasks.append({
                    "k": k,
                    "temperature": temperature,
                    "top_p": top_p,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "vectorstore": vectorstore
                })
        
        # === PHASE 1: ALL GENERATIONS ===
        if status_placeholder:
            status_placeholder.info(f"Phase 1/2: Running {len(tasks)} generation(s)...")
        
        # Initialize generator LLM once (use first task's params for initial, but we create chain per config)
        model_name = config["model_name"]
        generation_results = []
        
        for i, task in enumerate(tasks):
            try:
                # Create LLM with this task's temperature/top_p
                llm = get_llm(model_name, task["temperature"], task["top_p"])
                rag_chain = get_rag_chain(llm, task["vectorstore"], k=task["k"])
                
                gen_result = _run_generation(rag_chain, question, model_name)
                gen_result["task"] = task
                generation_results.append(gen_result)
                
                if status_placeholder:
                    status_placeholder.info(f"Phase 1/2: Generated {i+1}/{len(tasks)}...")
                    
            except Exception as e:
                generation_results.append({"successful": False, "error": str(e), "task": task})
        
        # Unload generator ONCE if Ollama
        if "Ollama" in model_name:
            if status_placeholder:
                status_placeholder.info("Unloading generator model...")
            unload_ollama_model(model_name)
        
        # === PHASE 2: ALL JUDGING ===
        if status_placeholder:
            status_placeholder.info(f"Phase 2/2: Running {len(tasks)} judgement(s)...")
        
        judge_model = config["judge_model"]
        # Initialize judge LLM once
        judge_llm = get_llm(judge_model, temperature=0.1)
        judge_chain = get_judge_chain(judge_llm)
        
        for i, gen_result in enumerate(generation_results):
            task = gen_result["task"]
            
            if not gen_result["successful"]:
                st.error(f"Skipping judge for failed generation (K={task['k']}): {gen_result.get('error')}")
                current_step += 1
                continue
            
            try:
                judge_result = _run_judging(
                    judge_chain, question, gen_result["answer"], gen_result["context"], judge_model
                )
                
                if judge_result["successful"]:
                    score = judge_result["score"]
                    results.append({
                        "Question": question,
                        "Answer": gen_result["answer"],
                        "Top-K": task["k"],
                        "Model": model_name,
                        "Judge": judge_model,
                        "Accuracy": score.get("accuracy"),
                        "Faithfulness": score.get("faithfulness"),
                        "Relevance": score.get("relevance"),
                        "Explanation": score.get("explanation"),
                        "Chunk Size": task["chunk_size"],
                        "Overlap": task["chunk_overlap"],
                        "Temperature": task["temperature"],
                        "Top P": task["top_p"],
                        "latency_rag": gen_result["latency_rag"],
                        "latency_judge": judge_result["latency_judge"]
                    })
                else:
                    st.error(f"Judge error (K={task['k']}): {judge_result.get('error')}")
                    
                if status_placeholder:
                    status_placeholder.info(f"Phase 2/2: Judged {i+1}/{len(generation_results)}...")
                    
            except Exception as e:
                st.error(f"Judge exception (K={task['k']}): {e}")
            
            current_step += 1
            if progress_bar:
                try:
                    progress_bar.progress(min(current_step / total_steps, 1.0))
                except: pass
        
        # Unload judge ONCE if Ollama
        if "Ollama" in judge_model:
            if status_placeholder:
                status_placeholder.info("Unloading judge model...")
            unload_ollama_model(judge_model)

    return results
