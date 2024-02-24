from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="models\codellama-7b-instruct.Q4_K_M.gguf",
    max_tokens=5000,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_batch=512,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
keep your answers simple and practical, if code been asked, provide the code files with the whole content.
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    input_variables=["question"],
    template=template,
)
# Chain
llm_chain = LLMChain(prompt=PROMPT, llm=llm)
question = """You are someone who has enough knowledge about python programming, email technologies (SMTP, POP3, IMAP, EWS, MAPI, DAV, WebDAV, HTTP), web applications development. email clients (Thunderbird, Outlook, Mozilla Thunderbird, Microsoft Outlook, Apple Mail, Mozilla Firefox, Google Chrome, Opera, Vivaldi).
you will be given an architecture, and reply with python solution.
Arch: data loader, record analyzer, and notification service.
Application will load data from csv file, analyze each record and set future notification for the user mentioned in the record. each record will have email and data, that notification should be scheduled to the next mounth metioned in the data."""
# Run
llm_response = llm_chain.invoke(question)