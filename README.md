## Code Llama: A REST Server Powered by Llama-cpp and Langchain

**Code Llama** is a Python application built on the **Langchain** framework that transforms the powerful **Llama-cpp** language model into a **RESTful API server**. This enables seamless integration with various tools and applications, allowing them to interact with the LLM through familiar API calls.

### Features

* **RESTful API:** Exposes Llama-cpp capabilities through a well-defined API, simplifying integration with existing tools and workflows.
* **Langchain Integration:** Leverages the flexibility and control of Langchain to tailor the API behavior and customize responses.
* **Llama-cpp Power:** Access the advanced text generation capabilities of Llama-cpp for diverse applications, including chatbots, Q&A systems, and text summarization.
* **Error Handling:** Ensures a smooth user experience by gracefully catching and reporting potential errors.

### Requirements

* Python 3.7+
* `fastapi`
* `pydantic`
* `langchain`
* `langchain-community`
* `llama-cpp-python`

### Files:
1. app.py # the server application.
2. exp.py # expirimantal code for using langchain llamacpp.
3. test.py # test file for testing the server.

### Installation

1. Clone this repository: `git clone https://github.com/your-username/code-llama`.
2. Install required libraries: `pip install -r requirements.txt`.
3. Download a supported Llama-cpp model and place it in the `models` directory (see [https://python.langchain.com/docs/integrations/llms/llamacpp](https://python.langchain.com/docs/integrations/llms/llamacpp)).

### Usage

1. **Start the server:** Run `uvicorn app:app --reload` in your terminal. This enables automatic code changes to be reflected without restarting the server during development (remove `--reload` for production).
2. **Send API requests:** Use tools like Postman or curl to send POST requests to the appropriate API endpoints, defined in the `app.py` file. Each endpoint typically has specific parameters and expected input formats. Refer to the endpoint documentation for details.

### Example

**Endpoint:** `/code-llama/chat/completions`

**Request:**

```json
POST /code-llama/chat/completions HTTP/1.1
Content-Type: application/json

{
  "model": "code-llama",
  "messages": [
    { "role": "user", "content": "What is the capital of France?" }
  ]
}
```

**Response:**

```json
{
  "id": 1234567890,
  "model": "code-llama",
  "created": 1677121600.0,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "bot",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {}
}
```