import requests

url = "http://127.0.0.1:8000/completion"  # Updated endpoint path

payload = {
    "prompt": "code simple hello world app using cpp",
    "n_predict": 5000,  # Adjust as needed
    "temperature": 0.1,  # Optional parameters
    "top_k": 40,
    "top_p": 0.95
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    print("Response received:")
    print(response.json())
else:
    print("Request failed with status code:", response.status_code)
    print("Error message:", response.text) 