import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

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
    
    target_model = "gemini-1.5-flash" # Default
    
    if "GPT-4" in model_name or "Sonnet" in model_name or "Pro" in model_name:
        target_model = "gemini-1.5-pro"
    
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
