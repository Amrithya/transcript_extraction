from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import yaml

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Loading configuration
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
            f"Transform the following extraction request into a JSON structure:\n"
            f"{request.extraction_info}\n"
            f"Provide only the JSON structure, without any explanation."
        )
        
        json_format_completion = client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI that creates JSON structures."},
                {"role": "user", "content": json_format_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        json_structure = json_format_completion.choices[0].message.content.strip()

        extraction_prompt = (
            f"Using the following JSON structure:\n{json_structure}\n"
            f"Extract the required information from this transcript and fill it into the JSON structure. "
            f"Provide only the filled JSON, without any explanation:\n{request.transcript}"
        )

        extraction_completion = client.chat.completions.create(
            model="llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "You are an AI that extracts information and formats it as JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        extracted_json = extraction_completion.choices[0].message.content.strip()

        # Check if the extracted JSON is not empty
        if not extracted_json:
            raise ValueError("Extracted JSON is empty")

        # Remove any leading/trailing whitespace or non-JSON characters
        extracted_json = extracted_json.strip()
        if extracted_json.startswith("```"):
            extracted_json = extracted_json[7:]
        if extracted_json.endswith("```"):
            extracted_json = extracted_json[:-3]

        # Parse the extracted JSON
        structured_info = json.loads(extracted_json)

        return structured_info

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}. Raw content: {extracted_json}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
