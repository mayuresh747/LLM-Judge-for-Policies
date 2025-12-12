from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models import BaseChatModel

class EvaluationScore(BaseModel):
    accuracy: int = Field(description="Score from 0-10 indicating how accurately the answer reflects the retrieved context.")
    faithfulness: int = Field(description="Score from 0-10 indicating if the answer is faithful to the context (no hallucinations).")
    relevance: int = Field(description="Score from 0-10 indicating how relevant the answer is to the user's question.")
    explanation: str = Field(description="A brief explanation of the scores.")

def get_judge_chain(llm: BaseChatModel):
    """
    Creates a chain that evaluates a RAG response.
    Input keys: question, answer, context
    """
    
    parser = JsonOutputParser(pydantic_object=EvaluationScore)
    
    prompt_str = """You are an expert AI evaluator.
    You will be given a user Question, a set of Retrieved Context, and a Generated Answer.
    
    Your task is to evaluate the Generated Answer on three criteria:
    1. Accuracy: Does the answer accurately reflect the facts in the context? (0-10)
    2. Faithfulness: Is the answer derived *only* from the context? Does it contain hallucinations? (0-10)
    3. Relevance: Does the answer directly address the user's question? (0-10)
    
    Return your evaluation in the following JSON format:
    {format_instructions}
    
    Question: {question}
    
    Retrieved Context:
    {context}
    
    Generated Answer:
    {answer}
    """
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_str,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    return chain
