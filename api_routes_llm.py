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

    def generate_structured_output(self, paragraph: str, transcript: str, output_schema: Dict[str, Any], max_attempts: int = 5):
        schema_str = json.dumps(output_schema, indent=2)

        prompt = f"""
        You are an expert information extraction assistant.
        Your task is to extract information from the provided transcript based on the given paragraph and return it in the specified JSON format.

        Paragraph describing the information to extract: {paragraph}

        Transcript: {transcript}

        Here is the output schema you must follow:

        {schema_str}

        Please provide a complete and valid JSON object that matches this schema without any additional text.
        """

        for attempt in range(max_attempts):
            try:
                with system():
                    self.model += "Generate a valid JSON object exactly matching the provided schema."

                with user():
                    self.model += prompt

                with assistant():
                    result = self.model + gen(
                        'output',
                        max_tokens=1024,
                        stop=None,
                        temperature=0.1
                    )

                json_str = result['output'].strip()
                parsed_output = json.loads(json_str)
                self.validate_output(parsed_output, output_schema)
                return parsed_output
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
            except ValueError as ve:
                logger.warning(f"Validation failed on attempt {attempt + 1}: {ve}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        raise ValueError(f"Failed to generate valid JSON after {max_attempts} attempts")

    def validate_output(self, output: Any, schema: Dict[str, Any]):
        if not isinstance(output, dict):
            raise ValueError("Output is not a dictionary")

        for key, value_type in schema.items():
            if key not in output:
                raise ValueError(f"Missing key: {key}")

            if isinstance(value_type, dict):
                self.validate_output(output[key], value_type)
            elif isinstance(value_type, list):
                if not isinstance(output[key], list):
                    raise ValueError(f"Expected list for key: {key}")
                for item in output[key]:
                    self.validate_output(item, value_type[0])
            elif value_type == "str" and not isinstance(output[key], str):
                raise ValueError(f"Expected string for key: {key}")

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

    with system():
        openai_model += prompt_func['system']
    with user():
        openai_model += prompt_func['user']
    with assistant():
        result = openai_model + gen('output', max_tokens=1024)

    transcript = result['output'].strip()
    return {"transcript": transcript}

def parse_paragraph(paragraph: str) -> List[str]:
    unified_paragraph = paragraph.replace("and", ",")
    return [field.strip() for field in unified_paragraph.split(",")]

# Defining route for information extraction
@app.post("/extract_information")
async def extract_information_endpoint(request: ExtractionRequest):
    logger.info(f"Received extraction request: {request}")

    fields_to_extract = parse_paragraph(request.paragraph)

    output_schema = {
        "extracted_info": {field: "str" for field in fields_to_extract}
    }

    generator = ConstrainedGenerator(openai_model)

    try:
        extracted_info = generator.generate_structured_output(
            paragraph=request.paragraph,
            transcript=request.transcript,
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
