import itertools
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from src.utils.ingestion import load_document, split_documents
from src.utils.vectorstore import create_vectorstore
from src.utils.llm_manager import get_llm
from src.utils.rag_chain import get_rag_chain
from src.utils.judge import get_judge_chain

def run_batch_experiment(
    file_path: str,
    config: Dict[str, Any],
    question: str,
    progress_bar: Any = None
) -> List[Dict[str, Any]]:
    """
    Runs a batch of experiments based on the configuration grid.
    
    Args:
        file_path: Path to the source document.
        config: The configuration dictionary containing lists for parameters.
        question: The user's question.
        progress_bar: Streamlit progress bar object (optional).
        
    Returns:
        A list of result dictionaries.
    """
    
    results = []
    
    # 1. Define Grid
    # Group by Ingestion Parameters (Chunk Size, Overlap) to optimize VectorStore creation
    ingestion_params = list(itertools.product(config["chunk_sizes"], config["chunk_overlaps"]))
    retrieval_params = config["k_retrievals"]
    
    total_steps = len(ingestion_params) * len(retrieval_params)
    current_step = 0
    
    # Pre-load document text (splitting happens later)
    # Note: load_document parses the file. If file type is markdown, we need to be careful.
    # We call load_document inside the loop? No, load once, split multiple times.
    raw_docs = load_document(file_path)
    
    for chunk_size, chunk_overlap in ingestion_params:
        
        # --- Ingestion Phase (Per Chunk Config) ---
        # Skip invalid configs where overlap >= chunk_size
        if chunk_overlap >= chunk_size:
            print(f"Skipping invalid config: Size={chunk_size}, Overlap={chunk_overlap}")
            # Update progress for skipped steps
            current_step += len(retrieval_params)
            if progress_bar:
                progress_bar.progress(current_step / total_steps)
            continue
            
        try:
            # Split
            chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Create VectorStore
            # This is the "expensive" part we want to do once per chunk config
            vectorstore = create_vectorstore(chunks)
            
        except Exception as e:
            st.error(f"Error during ingestion (Size={chunk_size}, Overlap={chunk_overlap}): {e}")
            current_step += len(retrieval_params)
            continue
            
        # --- Retrieval & Generation Phase ---
        for k in retrieval_params:
            try:
                # Setup Generator
                llm = get_llm(config["model_name"], config["temperature"], config["top_p"])
                
                # Setup Chain
                rag_chain = get_rag_chain(llm, vectorstore, k=k)
                
                # Run
                response = rag_chain.invoke({"input": question})
                answer = response["answer"]
                context_text = "\n\n".join([doc.page_content for doc in response["context"]])
                
                # Setup Judge
                judge_llm = get_llm(config["judge_model"], temperature=0.1)
                judge_chain = get_judge_chain(judge_llm)
                
                eval_input = {
                    "question": question,
                    "answer": answer,
                    "context": context_text
                }
                
                score = judge_chain.invoke(eval_input)
                
                # Record Result
                results.append({
                    "Question": question,
                    "Answer": answer, # Maybe truncate for display? No, users want to see it.
                    "Chunk Size": chunk_size,
                    "Overlap": chunk_overlap,
                    "Top-K": k,
                    "Model": config["model_name"],
                    "Judge": config["judge_model"],
                    "Accuracy": score.get("accuracy"),
                    "Faithfulness": score.get("faithfulness"),
                    "Relevance": score.get("relevance"),
                    "Explanation": score.get("explanation")
                })
                
            except Exception as e:
                st.error(f"Error during execution (K={k}): {e}")
            
            # Update Progress
            current_step += 1
            if progress_bar:
                progress_bar.progress(current_step / total_steps)
                
    return results
