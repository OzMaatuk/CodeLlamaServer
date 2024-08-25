import logging
from functools import lru_cache

from fastapi import FastAPI, APIRouter, Request, HTTPException
from pydantic import BaseModel, validator, Field
from typing import List, Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FastAPI app and router
app = FastAPI()
chat = APIRouter()

# --- LLM Configuration --- 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="models\codellama-7b-instruct.Q4_K_M.gguf",
    temperature=0.1,  # Use default temperature from server.py
    max_tokens=2048,     # Adjust max_tokens as needed
    n_ctx=2048*2,
    n_gpu_layers=-1,
    n_batch=512,
    f16_kv=True, 
    callback_manager=callback_manager,
    verbose=True,
)

# --- Pydantic Models ---
class CompletionRequest(BaseModel):
    prompt: str
    n_predict: Optional[int] = Field(None, description="Number of tokens to predict.")
    temperature: Optional[float] = Field(0.7, description="Temperature for sampling.")
    top_k: Optional[int] = Field(40, description="Top-k sampling.")
    top_p: Optional[float] = Field(0.95, description="Top-p (nucleus) sampling.")

class CompletionResponse(BaseModel):
    text: str

# --- Prompt Template ---
template = "{prompt}"  # No need for additional instructions for compatibility
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template=template,
)

# --- LLM Initialization (with lru_cache) ---
@lru_cache(maxsize=1)
def get_llm():
    """Returns the cached LlamaCpp model."""
    return llm

# --- Endpoint ---
@chat.post("/completion", response_model=CompletionResponse)
async def completion_endpoint(request: Request):
    """Generate text completions using the LlamaCpp model."""

    try:
        # Get request data
        request_data = await request.json()
        completion_request = CompletionRequest(**request_data)

        logging.info(f"Received completion request: {completion_request}")

        # Get the cached LLM
        llm = get_llm()

        # Set parameters on the llm object directly
        llm.temperature = completion_request.temperature
        llm.top_k = completion_request.top_k
        llm.top_p = completion_request.top_p
        n_predict = completion_request.n_predict if completion_request.n_predict else llm.max_tokens

        # Generate response using generate() 
        response = llm.generate([completion_request.prompt], max_tokens=n_predict)
        response_text = response.generations[0][0].text

        logging.info(f"LLM Response: {response_text}")

        return CompletionResponse(text=response_text)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")

app.include_router(chat)