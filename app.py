import streamlit as st
import pandas as pd
import tempfile
import os

from src.components.sidebar import render_sidebar
from src.utils.experiment import run_batch_experiment
from dotenv import load_dotenv
from openai import RateLimitError, InternalServerError

# Load env vars
load_dotenv()

st.set_page_config(page_title="RAG Eval Playground", layout="wide")

# Session State
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

st.title("RAG Evaluation Playground")

# Sidebar
config = render_sidebar()

# Main Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data Source")
    uploaded_files = st.file_uploader("Upload document(s) (PDF, TXT, MD)", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) ready for experiments.")

with col2:
    st.header("Playground")
    
    question = st.text_area("Enter your question:")
    
    if st.button("Run Experiment(s)"):
        if not question:
            st.error("Please enter a question.")
        else:
            # Save uploaded files to temp paths
            temp_file_paths = []
            
            # Create a temporary directory or just verify list
            # Note: users want to upload multiple, so we process them all as ONE context source
            # by merging them? Or iterating?
            # The requirement is "Allow to upload multiple files". 
            # In RAG, usually multiple files = larger knowledge base.
            # So we should pass ALL file paths to experiment, and it should ingest ALL.
            
            try:
                with st.spinner("Preparing Files..."):
                    # We need to keep files on disk for the loaders
                    # We can use a temp dir
                    t_dir = tempfile.mkdtemp()
                    for up_file in uploaded_files:
                        file_path = os.path.join(t_dir, up_file.name)
                        with open(file_path, "wb") as f:
                            f.write(up_file.getvalue())
                        temp_file_paths.append(file_path)

                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                status_placeholder.info("Step 1/3: Initializing models...")
                
                # Run Batch
                results = run_batch_experiment(
                    file_paths=temp_file_paths,
                    config=config,
                    question=question,
                    progress_bar=progress_bar,
                    status_placeholder=status_placeholder
                )
                
                # Append to history
                st.session_state.eval_results.extend(results)
                
                status_placeholder.empty()
                st.success(f"Completed {len(results)} runs!")
                
            except (RateLimitError, InternalServerError) as e:
                st.error(f"An API error occurred: {e}")
            except Exception as e:
                st.error(f"Experiment failed: {e}")
                
            finally:
                # Cleanup
                if 'temp_file_paths' in locals():
                    for p in temp_file_paths:
                        if os.path.exists(p):
                            os.remove(p)
                    if 't_dir' in locals() and os.path.exists(t_dir):
                        os.rmdir(t_dir)
                if 'progress_bar' in locals():
                    progress_bar.empty()
                


# Results Table
st.header("Experiment Results")
if st.session_state.eval_results:
    # Convert to DF
    df = pd.DataFrame(st.session_state.eval_results)
    
    # Reorder columns for better visibility
    cols = ["Question", "Answer", "Chunk Size", "Overlap", "Top-K", "Temperature", "Top P", "Accuracy", "Faithfulness", "Relevance", "Explanation", "latency_rag", "latency_judge"]
    
    # Rename columns for display
    df = df.rename(columns={"latency_rag": "Gen Time (s)", "latency_judge": "Judge Time (s)"})
    cols = ["Question", "Answer", "Chunk Size", "Overlap", "Top-K", "Temperature", "Top P", "Accuracy", "Faithfulness", "Relevance", "Explanation", "Gen Time (s)", "Judge Time (s)"]
    # Filter only columns that exist
    cols = [c for c in cols if c in df.columns]
    
    st.dataframe(df[cols])
    
    if st.button("Clear History"):
        st.session_state.eval_results = []
        st.rerun()
else:
    st.info("No results yet. Run an experiment!")
