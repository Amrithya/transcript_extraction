# Call Transcript Generator and Information Extractor

This project is a simple web application that allows users to generate call transcripts based on a given topic and then extract specific details from the generated transcript. It consists of a front-end built with HTML, CSS, and JavaScript, and a back-end API built using FastAPI.

## Features

- **Generate Call Transcript**: Users can input a topic, and the application will generate a call transcript using the OpenAI API.
- **Extract Information**: Users can specify details to extract from the generated transcript, and the application will return the relevant information in a structured format.

## Technologies Used

- **Front-End**: HTML, CSS, JavaScript
- **Back-End**: FastAPI, OpenAI API
- **Dependencies**:
  - `fastapi`
  - `pydantic`
  - `uvicorn`
  - `openai`
  - `python-dotenv`

## Getting Started

### Prerequisites

- Python 3.8 or higher
- A valid OpenAI API key
- Node.js (for serving the static files if needed)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Amrithya/transcript_extraction
   cd transcript_extraction
   
2. Install the dependencies:

   ```bash
   pip install uv
   uv venv ENV_NAME # Creating a clean environment and handling dependencies
   YOUR_ENV_PATH/../Scripts/activate
   uv pip install -r requirements.txt
     
3. Add your OpenAI API key:

   Use scaleway Generative APIs 
   https://www.scaleway.com/en/docs/ai-data/generative-apis/quickstart/

4. Create an .env File:

   In your project directory, create a .env file (you can do this using a text editor or the command line).
   Add your environment variables to the .env file in the following format:

   ```bash
   access_key: YOUR_ACCESS_KEY
   default_organization_id: YOUR_ORGANIZATION_ID
   default_project_id: YOUR_PROJECT_ID
   secret_key: YOUR_SECRET_KEY

5. Test your connections using the test files: 

   Update the secret_key and service url in the test files before running

   ```bash
      uv run YOUR_FILE_NAME.py


6. Run the FastAPI server:
   
   ```bash
   uv run api_routes_llm.py
   
7. Open a web browser and navigate :

   ```bash
   ui/transcript_extraction.html





