# https://python.langchain.com/docs/integrations/llms/llamacpp
import datetime
import random

from fastapi import FastAPI, APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# Create an APIRouter instance
app = FastAPI()  # Create the application instance
chat = APIRouter()

# Create llm model 
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="models\codellama-7b.Q4_K_M.gguf",
    temperature=0.1,
    max_tokens=5000,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_batch=512,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

# Pydantic models (no changes needed here)
class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    # Add model properties as needed
    pass

class ResponseData(BaseModel):
    id: int
    model: str
    created: float
    choices: List[Choice]
    usage: Usage

class RequestData(BaseModel):
    model: str
    messages: List[Message]

# Gemini API client initialization (updated model name)
def get_llm_client():
    # Prompt
    template = """Aanswer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep your answers simple and practical, dont answer anything else.
    If code been asked, provide the code files with the whole content and no isolated code blocks.
    If code fix is requiered, provide the full code version as your answer and not just the relevant fixes.
    You need to provide only one accurate answer.
    Question: {question}
    Helpful Answer:"""
    PROMPT = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    # Chain
    return LLMChain(prompt=PROMPT, llm=llm)

# API endpoint using Gemini API (updated method call)
@chat.post("/code-llama/chat/completions", response_model=ResponseData, summary="Send a chat message to Gemini and get an OpenAI-compatible API response.")
async def chat_completions_endpoint(request: RequestData):
    try:
        model = request.model
        messages = request.messages
        system_message = "system_message: "

        # Prepare prompt for Gemini request
        prompt = f"{system_message}\n" + ''.join([f"{m.role}: {m.content}\n" for m in messages])

        # Send request to Gemini API using Python client (updated method call)
        llm_client = get_llm_client()
        response_text = str(llm_client.invoke(prompt))

        # Create response using the model (no changes needed)
        response_dict = ResponseData(
            id=random.randint(1, 999999999999999999999),
            model=model,
            created=datetime.datetime.now().timestamp(),
            choices=[
                Choice(
                    index=0,
                    message=Message(role="bot", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage()
        )
        return response_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail="Server Error: " + str(e))

app.include_router(chat)  # Mount the chat APIRouter to the main app