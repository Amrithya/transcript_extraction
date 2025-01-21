# Importing necessary modules and classes
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import tiktoken
from guidance import models, gen, system, user, assistant
import logging
import json
import re

# Creating FastAPI application instance
app = FastAPI()

# Mounting static files for UI
app.mount("/ui", StaticFiles(directory="./ui"), name="static")

# Adding CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setting up tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Configuring OpenAI model
openai_model = models.OpenAI(   
    model="llama-3.1-8b-instruct",
    max_streaming_tokens=1024,
    tokenizer=tokenizer,
    base_url="https://api.scaleway.ai/43712c54-4844-4d14-a045-6ebe90fb5221/v1",
    echo=False
)

# Defining request models
class TopicRequest(BaseModel):
    topic: str

class ExtractionRequest(BaseModel):
    paragraph: str
    transcript: str

# Implementing constrained generator class
class ConstrainedGenerator:
    def __init__(self, model):
        self.model = model

    # Generating structured output based on context and schema
    def generate_structured_output(self, context: str, output_schema: Dict[str, Any], max_attempts: int = 5):
        schema_str = json.dumps(output_schema, indent=2)
        
        prompt = f"""
        You are an expert information extraction assistant. 
        Extract information following this exact JSON structure:
        
        {schema_str}
        
        Context: {context}
        
        Output the extracted information in the specified JSON format. 
        Ensure your response is a complete, valid JSON object.
        Do not include any text before or after the JSON object.
        """
        
        for attempt in range(max_attempts):
            try:
                # Generating response using the model
                with system():
                    self.model += "Generate a valid JSON object exactly matching the provided schema. No additional text."
                
                with user():
                    self.model += prompt
                
                with assistant():
                    result = self.model + gen(
                        'output', 
                        max_tokens=1024, 
                        stop=None,
                        temperature=0.1
                    )
                
                # Processing and parsing the generated JSON
                json_str = result['output'].strip()
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = json_str[start:end]
                
                json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
                
                parsed_output = json.loads(json_str)
                return parsed_output
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                logger.warning(f"Generated output: {json_str}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        raise ValueError(f"Failed to generate valid JSON after {max_attempts} attempts")

# Defining route for root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("transcript_extraction.html", "r") as f:
        return f.read()

# Defining route for transcript generation
@app.post("/generate_transcript")
async def generate_transcript(request: TopicRequest):
    global openai_model
    topic = request.topic
    
    prompt_func = {
        'system': "You will generate a call transcript based on the topic, and it can vary in structure.",
        'user': f"Generate a call transcript between a client and a contact center agent discussing {topic}."
    }
    
    # Generating transcript using the model
    with system():
        openai_model += prompt_func['system']
    with user():
        openai_model += prompt_func['user']
    with assistant():
        result = openai_model + gen('output', max_tokens=1024)
    
    transcript = result['output'].strip()
    return {"transcript": transcript}

# Defining route for information extraction
@app.post("/extract_information")
async def extract_information_endpoint(request: ExtractionRequest):
    logger.info(f"Received extraction request: {request}")
    
    output_schema = {
        "issues": [
            {
                "description": "str"
            }
        ],
        "solution": "str",
        "additional_info": {
            "key": "str"
        }
    }
    
    generator = ConstrainedGenerator(openai_model)
    
    try:
        # Extracting information using the constrained generator
        extracted_info = generator.generate_structured_output(
            context=request.transcript, 
            output_schema=output_schema
        )
        return JSONResponse(content=extracted_info)
    
    except ValueError as e:
        logger.error(f"Extraction error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Failed to extract information. Please try again."}
        )

# Defining global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."}
    )

# Running the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
