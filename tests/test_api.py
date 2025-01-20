from openai import OpenAI
from dotenv import load_dotenv, dotenv_values 
import requests
import json
import os

load_dotenv()  # Loading environment variables from .env file
api_key = os.getenv("secret_key")

# Initialize OpenAI client with Scaleway's base URL
client = OpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=api_key
)

URL = YOUR_SERVICE_URL

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer $$YOUR_SECRET_KEY$$"  # Remove $ sign
}

PAYLOAD = {
    "model": "llama-3.1-8b-instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
		{ "role": "user", "content": "Describe a futuristic city with advanced technology and green energy solutions for sustainability with people." },
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 1,
    "presence_penalty": 0,
    "stream": True,
}

response = requests.post(URL, headers=HEADERS, json=PAYLOAD, stream=True)

for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8').strip()
        if decoded_line == "data: [DONE]":
            break
        if decoded_line.startswith("data: "):
            try:
                data = json.loads(decoded_line[len("data: "):])
                if data.get("choices") and data["choices"][0]["delta"].get("content"):
                    print(data["choices"][0]["delta"]["content"], end="")
            except json.JSONDecodeError:
                continue

def translate_text(input_str):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are an expert translator who translates text from English to French and only returns translated text",
            },
            {"role": "user", "content": input_str},
        ],
    )

    return completion.choices[0].message.content

# Test the function
print(translate_text("This is a test string to translate"))
