import os
import requests
import subprocess
import time
import shutil
import streamlit as st # For logging/warning
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel

def is_ollama_running() -> bool:
    """Checks if Ollama is reachable at localhost:11434."""
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_server() -> bool:
    """Attempts to start the Ollama server."""
    if is_ollama_running():
        return True
        
    print("Ollama not running. Attempting to start...")
    try:
        # Check if ollama is in PATH
        if not shutil.which("ollama"):
            st.error("Ollama executable not found in PATH.")
            return False

        # Start process
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for startup
        for _ in range(10): # Wait up to 10s
            time.sleep(1)
            if is_ollama_running():
                print("Ollama started successfully.")
                return True
                
        st.error("Timed out waiting for Ollama to start.")
        return False
        
    except Exception as e:
        st.error(f"Failed to start Ollama: {e}")
        return False
        
def ensure_ollama_reachable() -> bool:
    """Ensures Ollama is running, starting it if necessary."""
    if is_ollama_running():
        return True
    return start_ollama_server()

def unload_ollama_model(model_name: str) -> bool:
    """
    Unloads an Ollama model from memory by setting keep_alive to 0.
    This frees VRAM/RAM for the next model.
    """
    # Map friendly names to Ollama model IDs
    model_id = "mistral"  # Default
    if "Llama 3.2" in model_name:
        model_id = "llama3.2"
    elif "Mistral" in model_name:
        model_id = "mistral"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_id,
                "prompt": "",
                "keep_alive": 0
            },
            timeout=10
        )
        if response.status_code == 200:
            print(f"Unloaded model: {model_id}")
            return True
        else:
            print(f"Failed to unload model: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error unloading model: {e}")
        return False

def get_llm(model_name: str, temperature: float = 0.7, top_p: float = 0.9) -> BaseChatModel:
    """
    Returns a configured LLM instance.
    
    Args:
        model_name: The name of the model selected in the UI.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        
    Returns:
        A LangChain BaseChatModel.
    """
    
    # Check for Groq Models
    if "Groq" in model_name:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("Missing GROQ_API_KEY. Check .env file.")
        
        # Map friendly names to Groq Model IDs (Using Llama 3.1)
        target_model = "llama-3.1-8b-instant" # Default
        
        if "70b" in model_name:
            target_model = "llama-3.1-70b-versatile"
        elif "Mixtral" in model_name:
            target_model = "mixtral-8x7b-32768"
        elif "Llama 3" in model_name: # Fallback
             # Don't overwrite if it's 8b (which is default), but if ambiguous use 70b?
             # Actually, better to just stick to default "llama-3.1-8b-instant" unless specified.
             pass 
            
        return ChatGroq(
            model=target_model,
            temperature=temperature,
            groq_api_key=api_key,
            model_kwargs={"top_p": top_p}
        )

    # Check for GitHub Models
    if "GitHub" in model_name:
        base_url = "https://models.inference.ai.azure.com"
        api_key = None
        target_model = None
        
        if "Grok 3" in model_name:
            api_key = os.getenv("GITHUB_TOKEN_GROK")
            target_model = "grok-3" # Assumption: Model ID is grok-3
        elif "GPT-4o" in model_name:
            api_key = os.getenv("GITHUB_TOKEN_OPENAI")
            target_model = "gpt-4o"
            
        if not api_key:
            st.error(f"Missing API Token for {model_name}. Check .env file.")
            # Fallback or error? defaulting to error will show in UI
        
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            model_kwargs={"top_p": top_p}
        )

    # Check for Ollama Models
    if "Ollama" in model_name:
        target_model = "mistral" # Default
        if "Llama 3.2" in model_name:
            target_model = "llama3.2"
        elif "Mistral" in model_name:
            target_model = "mistral"
            
        return ChatOllama(
            model=target_model,
            temperature=temperature,
            base_url="http://localhost:11434",
            # Ollama doesn't strictly adhere to top_p in the same kwarg structure sometimes 
            # but langchain handles it or ignores it. 
            top_p=top_p 
        )

    # Legacy Google/Gemini Logic
    # Map UI selections to available Gemini models
    # Since we are forced to use Gemini keys, we will map "GPT-4o", etc. to Gemini models
    # but perhaps with different system prompts or just logging the intent.
    # Ideally, we should use the requested model if keys were available.
    # Given the constraint: "use gemini keys you have for now"
    
    # We will map everything to gemini-pro or gemini-1.5-pro for now, 
    # but we can try to respect the naming if we were using a different provider.
    
    # For this implementation, we will use 'gemini-1.5-flash' as a default fast model
    # and 'gemini-1.5-pro' as a more capable model.
    
    target_model = "gemini-flash-latest" # Default
    
    if "GPT-4" in model_name or "Sonnet" in model_name or "Pro" in model_name:
        target_model = "gemini-pro-latest"
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fallback for UI testing without keys, though it will fail on execution
        print("Warning: GOOGLE_API_KEY not set.")
    
    llm = ChatGoogleGenerativeAI(
        model=target_model,
        temperature=temperature,
        top_p=top_p,
        google_api_key=api_key,
        convert_system_message_to_human=True # Sometimes needed for older Gemini, usually fine now
    )
    
    return llm
