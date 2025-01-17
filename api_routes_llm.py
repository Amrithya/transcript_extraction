from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import yaml

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the front-end URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Load configuration
with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)

api_key = data['secret_key']

client = OpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=api_key
)

class Topic(BaseModel):
    topic: str = Field(..., min_length=3, max_length=100, description="Topic for generating transcript")

class ExtractionRequest(BaseModel):
    extraction_info: str = Field(..., min_length=10, description="Paragraph describing the information to extract")
    transcript: str = Field(..., min_length=50, description="Call transcript")

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("ui/transcript_extraction.html") as file:
        return file.read()

@app.post("/generate_transcript")
async def generate_transcript(topic: Topic):
    if not topic.topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI that generates realistic call transcripts between a client and a contact center agent."},
                {"role": "user", "content": f"Generate a call transcript about {topic.topic}."}
            ],
            temperature=0.7,
            max_tokens=500
        )

        transcript = completion.choices[0].message.content

        return {"transcript": transcript}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_information")
async def extract_information(request: ExtractionRequest):
    try:
        json_format_prompt = (
            f"Transform the following extraction request into a JSON-izable format:\n"
            f"{request.extraction_info}"
        )
        
        json_format_completion = client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI that formats information into JSON."},
                {"role": "user", "content": json_format_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        json_structure = json_format_completion.choices[0].message.content.strip()

        extraction_prompt = (
            f"Using the following JSON structure:\n{json_structure}\n\n"
            f"Extract the required information from this transcript:\n{request.transcript}"
        )

        extraction_completion = client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI that extracts information based on a JSON structure."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        extracted_info = extraction_completion.choices[0].message.content.strip()

        structured_info = {}
        for line in extracted_info.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                structured_info[key.strip()] = value.strip()

        return structured_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
