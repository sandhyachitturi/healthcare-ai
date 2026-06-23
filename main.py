from fastapi import FastAPI
from pydantic import BaseModel, Field
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
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

# Simple in-memory store — NOTE: resets every time the server restarts.
# Day 5 goal is to understand the PATTERN, not build production storage yet.
conversation_store: dict[str, list[dict]] = {}

class ClinicalQuery(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=2000)

class ClinicalResponse(BaseModel):
    answer: str
    model_used: str
    question_received: str
    session_id: str

@app.post("/ask", response_model=ClinicalResponse)
async def ask_clinical(query: ClinicalQuery):
    # Load this session's past turns (or start fresh)
    history = conversation_store.get(query.session_id, [])

    # Add the new user message to the running history
    history.append({"role": "user", "content": query.question})

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=CLINICAL_SYSTEM_PROMPT,
        messages=history  # send the FULL history, not just this one question
    )

    answer_text = message.content[0].text

    # Save Claude's reply into the history too, so the next turn has it
    history.append({"role": "assistant", "content": answer_text})
    conversation_store[query.session_id] = history

    return ClinicalResponse(
        answer=answer_text,
        model_used="claude-sonnet-4-6",
        question_received=query.question,
        session_id=query.session_id
    )