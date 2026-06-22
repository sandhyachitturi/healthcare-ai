from fastapi import FastAPI
from pydantic import BaseModel, Field
import anthropic
import os
from dotenv import load_dotenv

# Load your API key from .env file
load_dotenv()

# Create the FastAPI app
app = FastAPI()

# Create the Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Request model — what comes IN
class ClinicalQuery(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

# Response model — what goes OUT
class ClinicalResponse(BaseModel):
    answer: str
    model_used: str
    question_received: str

# The endpoint
@app.post("/ask", response_model=ClinicalResponse)
async def ask_clinical(query: ClinicalQuery):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="You are a clinical decision support assistant.",
        messages=[{"role": "user", "content": query.question}]
    )
    return ClinicalResponse(
        answer=message.content[0].text,
        model_used="claude-sonnet-4-6",
        question_received=query.question
    )