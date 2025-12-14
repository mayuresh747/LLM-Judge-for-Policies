import streamlit as st

def render_sidebar():
    """Renders the sidebar and returns the configuration."""
    
    st.sidebar.header("Configuration")
    
    # Model Selection
    st.sidebar.subheader("Generator Settings")
    generator_options = ["Llama 3.1 70b (Groq)", "Llama 3.1 8b (Groq)", "Mixtral 8x7b (Groq)", "Mistral (Ollama)", "Llama 3.2 (Ollama)", "Gemini Flash (Latest)", "Gemini Pro (Latest)", "Grok 3 (GitHub)", "GPT-4o (GitHub)", "GPT-4o", "Claude 3.5 Sonnet", "Llama-3"]
    selected_model = st.sidebar.selectbox("Select Generator Model", generator_options, index=0)
    
    # Judge Selection
    st.sidebar.subheader("Judge Settings")
    judge_options = ["Llama 3.1 70b (Groq)", "Llama 3.1 8b (Groq)", "Mixtral 8x7b (Groq)", "Mistral (Ollama)", "Llama 3.2 (Ollama)", "Gemini Flash (Latest)", "Gemini Pro (Latest)", "Grok 3 (GitHub)", "GPT-4o (GitHub)", "GPT-4o", "Claude 3.5 Sonnet", "Llama-3"]
    selected_judge = st.sidebar.selectbox("Select Judge Model", judge_options, index=0)
    
    # Mode Toggle
    mode = st.sidebar.radio("Mode", ["Parameter Tuning", "RAG Evaluation", "Combined"])
    
    # Concurrency
    max_concurrency = st.sidebar.slider("Concurrency (Max Threads)", 1, 10, 1, help="Limit parallel requests to avoid Rate Limits.")
    
    # Parameters
    st.sidebar.subheader("Model Parameters")
    
    if mode in ["Parameter Tuning", "Combined"]:
        # Multi-select for tuning/combined
        temperatures = st.sidebar.multiselect(
            "Temperatures", 
            [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 
            default=[0.1, 0.7]
        )
        top_ps = st.sidebar.multiselect(
            "Top P Values", 
            [0.1, 0.5, 0.9, 1.0], 
            default=[0.9]
        )
        # Ensure defaults
        if not temperatures: temperatures = [0.7]
        if not top_ps: top_ps = [0.9]
        
    else:
        # Single value for RAG Eval (but stored as list for consistency)
        temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        tp = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.1)
        temperatures = [temp]
        top_ps = [tp]
    
    # RAG Parameters
    st.sidebar.subheader("RAG Parameters")
    
    if mode in ["RAG Evaluation", "Combined"]:
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
        # Parameter Tuning Mode (Single Value for RAG params)
        chunk_sizes = [st.sidebar.number_input("Chunk Size", 100, 2000, 1000, 100)]
        chunk_overlaps = [st.sidebar.number_input("Chunk Overlap", 0, 500, 200, 50)]
        k_retrievals = [st.sidebar.slider("Top-K Retrieval", 1, 10, 3)]
    
    config = {
        "model_name": selected_model,
        "judge_model": selected_judge,
        "mode": mode,
        "max_concurrency": max_concurrency,
        "temperatures": temperatures,     # List
        "top_ps": top_ps,                 # List
        "chunk_sizes": chunk_sizes,       # List
        "chunk_overlaps": chunk_overlaps, # List
        "k_retrievals": k_retrievals      # List
    }
    
    # Cost Estimation Display
    total_combinations = len(chunk_sizes) * len(chunk_overlaps) * len(k_retrievals) * len(temperatures) * len(top_ps)
    total_calls = total_combinations * 2 # Generator + Judge
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Safety Guardrails")
    st.sidebar.write(f"**Total Experiment Runs:** {total_combinations}")
    st.sidebar.write(f"**Est. API Calls:** {total_calls}")
    
    if total_calls > 10:
        st.sidebar.error(f"⚠️ High usage! This will consume {total_calls} calls. Daily limit is ~15.")
    else:
        st.sidebar.success(f"✅ Safe. {total_calls} calls within typical daily limits.")
    
    return config
