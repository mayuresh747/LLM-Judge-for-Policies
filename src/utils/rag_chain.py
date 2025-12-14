from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
from datetime import date
import os

def load_system_prompt() -> str:
    """Loads the system prompt from config file and replaces {{today}} with current date."""
    config_path = os.path.join(os.path.dirname(__file__), "../../config/system_prompt.txt")
    try:
        with open(config_path, "r") as f:
            prompt = f.read()
        # Replace date placeholder
        prompt = prompt.replace("{{today}}", date.today().strftime("%Y-%m-%d"))
        return prompt
    except FileNotFoundError:
        print(f"Warning: System prompt not found at {config_path}. Using default.")
        return "You are a helpful assistant. Answer the user's question."

def get_rag_chain(llm: BaseChatModel, vectorstore: Optional[VectorStore], k: int = 3) -> Runnable:
    """
    Creates a RAG chain given an LLM and a VectorStore.
    If vectorstore is None, returns a simple LLM chain.
    """
    
    system_prompt_text = load_system_prompt()
    
    if vectorstore is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("human", "{input}"),
        ])
        
        # Simple chain: prompt -> llm -> str_parser
        chain = prompt | llm | StrOutputParser()
        
        # Wrapper to match RAG output format: {"answer": ..., "context": []}
        def format_output(answer):
            return {"answer": answer, "context": []}
            
        return chain | RunnableLambda(format_output)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # Append context placeholder to system prompt for RAG
    rag_system_prompt = system_prompt_text + "\n\n---------------------------------------------------------------------\nRETRIEVED CONTEXT\n---------------------------------------------------------------------\n{context}"
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

