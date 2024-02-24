import requests

url = "http://127.0.0.1:8000/code-llama/chat/completions/"  # Replace with your server's URL

payload = {
    "model": "code-llama",
    "messages": [
        { "role": "user", "content": "please create simple notification service, consider abstract class and (read from db, calculate if to notify, etc...) and implementation for the email feature. write 'stop' when you done." }
    ]
}

headers = {
    "Content-Type": "application/json"  # No need for Authorization header now
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    print("Response received:")
    print(response.json())
else:
    print("Request failed with status code:", response.status_code)
    print("Error message:", response.text)