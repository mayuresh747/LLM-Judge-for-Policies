import streamlit as st

def render_sidebar():
    """Renders the sidebar and returns the configuration."""
    
    st.sidebar.header("Configuration")
    
    # Model Selection
    st.sidebar.subheader("Generator Settings")
    generator_options = ["Grok 3 (GitHub)", "GPT-4o (GitHub)", "GPT-4o", "Claude 3.5 Sonnet", "Llama-3", "Gemini 1.5 Pro"]
    selected_model = st.sidebar.selectbox("Select Generator Model", generator_options, index=0)
    
    # Judge Selection
    st.sidebar.subheader("Judge Settings")
    judge_options = ["Grok 3 (GitHub)", "GPT-4o (GitHub)", "GPT-4o", "Claude 3.5 Sonnet", "Llama-3", "Gemini 1.5 Pro"]
    selected_judge = st.sidebar.selectbox("Select Judge Model", judge_options, index=1)
    
    # Mode Toggle
    mode = st.sidebar.radio("Mode", ["Parameter Tuning", "RAG Evaluation"])
    
    # Parameters
    st.sidebar.subheader("Model Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.1)
    
    # RAG Parameters
    st.sidebar.subheader("RAG Parameters")
    
    if mode == "RAG Evaluation":
        st.sidebar.info("Select multiple options for Grid Search")
        chunk_sizes = st.sidebar.multiselect(
            "Chunk Sizes", 
            [100, 250, 500, 1000, 2000], 
            default=[500, 1000]
        )
        chunk_overlaps = st.sidebar.multiselect(
            "Chunk Overlaps", 
            [0, 50, 100, 200], 
            default=[50, 100]
        )
        k_retrievals = st.sidebar.multiselect(
            "Top-K Retrievals", 
            [1, 3, 5, 7, 10], 
            default=[3, 5]
        )
        
        # Ensure at least one value is selected to avoid errors
        if not chunk_sizes: chunk_sizes = [1000]
        if not chunk_overlaps: chunk_overlaps = [200]
        if not k_retrievals: k_retrievals = [3]
        
    else:
        # Parameter Tuning Mode (Single Value)
        chunk_sizes = [st.sidebar.number_input("Chunk Size", 100, 2000, 1000, 100)]
        chunk_overlaps = [st.sidebar.number_input("Chunk Overlap", 0, 500, 200, 50)]
        k_retrievals = [st.sidebar.slider("Top-K Retrieval", 1, 10, 3)]
    
    config = {
        "model_name": selected_model,
        "judge_model": selected_judge,
        "mode": mode,
        "temperature": temperature,
        "top_p": top_p,
        "chunk_sizes": chunk_sizes,       # List
        "chunk_overlaps": chunk_overlaps, # List
        "k_retrievals": k_retrievals      # List
    }
    
    return config
