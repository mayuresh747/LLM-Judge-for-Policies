import streamlit as st
import pandas as pd
import tempfile
import os

from src.components.sidebar import render_sidebar
from src.utils.experiment import run_batch_experiment
from dotenv import load_dotenv

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
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    
    # We no longer "Process" immediately in batch mode because processing depends on the experiment config.
    # However, for UX, maybe we just store the file path in session state after upload.
    # Since Streamlit re-runs, we need to handle the file carefully.
    
    if uploaded_file:
        st.info(f"File '{uploaded_file.name}' ready for experiments.")

with col2:
    st.header("Playground")
    
    question = st.text_area("Enter your question:")
    
    if st.button("Run Experiment(s)"):
        if not uploaded_file:
            st.error("Please upload a document.")
        elif not question:
            st.error("Please enter a question.")
        else:
            # Save uploaded file to temp path
            # We do this every run to ensure freshness/simplicity
            with st.spinner("Preparing Experiment..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

            try:
                progress_bar = st.progress(0)
                st.write("Running experiments... This may take a while depending on the grid size.")
                
                # Run Batch
                results = run_batch_experiment(
                    file_path=tmp_path,
                    config=config,
                    question=question,
                    progress_bar=progress_bar
                )
                
                # Append to history
                st.session_state.eval_results.extend(results)
                
                st.success(f"Completed {len(results)} runs!")
                
            except Exception as e:
                st.error(f"Experiment failed: {e}")
                
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                progress_bar.empty()

# Results Table
st.header("Experiment Results")
if st.session_state.eval_results:
    # Convert to DF
    df = pd.DataFrame(st.session_state.eval_results)
    
    # Reorder columns for better visibility
    cols = ["Question", "Answer", "Chunk Size", "Overlap", "Top-K", "Accuracy", "Faithfulness", "Relevance", "Explanation"]
    # Filter only columns that exist
    cols = [c for c in cols if c in df.columns]
    
    st.dataframe(df[cols])
    
    if st.button("Clear History"):
        st.session_state.eval_results = []
        st.rerun()
else:
    st.info("No results yet. Run an experiment!")
