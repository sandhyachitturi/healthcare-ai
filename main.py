from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import anthropic
import logging
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthcare-ai")

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
    history = conversation_store.get(query.session_id, [])
    history.append({"role": "user", "content": query.question})

    try:
        logger.info(f"Session {query.session_id}: sending question to Claude")
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=CLINICAL_SYSTEM_PROMPT,
            messages=history
        )
    except anthropic.AuthenticationError:
        logger.error("Authentication failed — check ANTHROPIC_API_KEY")
        raise HTTPException(status_code=500, detail="Server configuration error. Please contact support.")
    except anthropic.APIConnectionError:
        logger.error("Could not connect to Anthropic API")
        raise HTTPException(status_code=503, detail="AI service is temporarily unavailable. Please try again shortly.")
    except anthropic.RateLimitError:
        logger.warning("Rate limit hit")
        raise HTTPException(status_code=503, detail="Service is busy. Please try again in a moment.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong processing your request.")

    answer_text = message.content[0].text
    history.append({"role": "assistant", "content": answer_text})
    conversation_store[query.session_id] = history

    logger.info(f"Session {query.session_id}: response sent successfully")

    return ClinicalResponse(
        answer=answer_text,
        model_used="claude-sonnet-4-6",
        question_received=query.question,
        session_id=query.session_id
    )
 # Minimal HTML test client — visit http://localhost:8000/
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head><title>Clinical AI Assistant</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h2>Clinical Decision Support Assistant</h2>
        <input id="question" type="text" placeholder="Ask a clinical question..." style="width: 100%; padding: 8px;">
        <button onclick="ask()" style="margin-top: 8px; padding: 8px 16px;">Ask</button>
        <p id="answer" style="margin-top: 20px; white-space: pre-wrap;"></p>

        <script>
        async function ask() {
            const question = document.getElementById('question').value;
            document.getElementById('answer').innerText = "Thinking...";
            const res = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: 'demo-session', question: question})
            });
            if (!res.ok) {
                const err = await res.json();
                document.getElementById('answer').innerText = "Error: " + err.detail;
                return;
            }
            const data = await res.json();
            document.getElementById('answer').innerText = data.answer;
        }
        </script>
    </body>
    </html>
    """