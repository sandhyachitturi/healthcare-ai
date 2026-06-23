from fastapi import FastAPI
from pydantic import BaseModel, Field
import anthropic
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# ASYNC client — note AsyncAnthropic, not Anthropic
client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CLINICAL_SYSTEM_PROMPT = """You are a clinical decision support assistant for healthcare providers.

Your role:
- Answer questions about conditions, medications, symptoms, and clinical guidelines
- Provide information that helps clinicians make informed decisions
- Be clear, concise, and clinically accurate

Your boundaries:
- You do not diagnose individual patients or replace clinical judgment
- If a question is outside healthcare/clinical topics, politely redirect and explain your scope
- If you lack sufficient information to answer safely, say so explicitly rather than guessing
- Always note that your output supports, but does not replace, professional clinical judgment

Format your answers clearly, using brief structure (like short lists) when it improves readability for a busy clinician."""

class ClinicalQuery(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

class ClinicalResponse(BaseModel):
    answer: str
    model_used: str
    question_received: str

@app.post("/ask", response_model=ClinicalResponse)
async def ask_clinical(query: ClinicalQuery):
    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=CLINICAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query.question}]
    )
    return ClinicalResponse(
        answer=message.content[0].text,
        model_used="claude-sonnet-4-6",
        question_received=query.question
    )

# Proof-of-concept endpoint — simulates a slow task
@app.get("/slow-test")
async def slow_test():
    await asyncio.sleep(5)
    return {"message": "Done waiting 5 seconds"}