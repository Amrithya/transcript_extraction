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
  - `pyyaml`

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
   pip install -r requirements.txt
     
3. Add your OpenAI API key to config.yaml:

   Use scaleway Generative APIs 
   https://www.scaleway.com/en/docs/ai-data/generative-apis/quickstart/

4. Run the FastAPI server:
   
   ```bash
   uvicorn api_routes_llm:app
   
5. Open a web browser and navigate :

   ```bash
   ui/transcript_extraction.html
  
















