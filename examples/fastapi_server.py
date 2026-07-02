"""Serve the RL-augmented sales agent as a RESTful API with FastAPI.

    pip install "rl-sales-augment[gemini,api]"       # or [openai] / [anthropic] + [api]
    uvicorn fastapi_server:app --host 0.0.0.0 --port 8000

The endpoint is stateless in the ChatGPT style: the client sends the FULL conversation as
OpenAI-format messages each call; the agent rebuilds its belief from the history, the RL policy
picks the move, the LLM writes the reply.

    curl -X POST localhost:8000/v1/chat -H 'Content-Type: application/json' -d '{
      "messages": [
        {"role": "user", "content": "hey, what does NimbusBox cost?"},
        {"role": "assistant", "content": "Depends on the setup. What are you running today?"},
        {"role": "user", "content": "a few old racks. honestly budget is tight this quarter"}
      ],
      "segment": 7
    }'
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rl_sales_augment as rsa

COMPANY_CTX = os.environ.get("RSA_COMPANY_CTX", "")   # your company facts block

def make_generate_fn():
    """Pick the LLM from env: GCP_PROJECT (Gemini/Vertex ADC), OPENAI_API_KEY, or ANTHROPIC_API_KEY."""
    if os.environ.get("GCP_PROJECT"):
        return rsa.providers.gemini_vertex(project=os.environ["GCP_PROJECT"])
    if os.environ.get("OPENAI_API_KEY"):
        return rsa.providers.openai_chat(model=os.environ.get("RSA_MODEL", "gpt-5.5"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        return rsa.providers.anthropic_chat(model=os.environ.get("RSA_MODEL", "claude-sonnet-5"))
    raise RuntimeError("set GCP_PROJECT, OPENAI_API_KEY, or ANTHROPIC_API_KEY")

app = FastAPI(title="rl-sales-augment API", version=rsa.__version__)
_bots: dict = {}                                       # one agent per segment (tiny; CPU)

def get_bot(segment: Optional[int]):
    key = segment if segment is not None else -1
    if key not in _bots:
        _bots[key] = rsa.load_agent(make_generate_fn(), company_ctx=COMPANY_CTX, segment=segment)
    return _bots[key]

class Message(BaseModel):
    role: str                                          # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    segment: Optional[int] = None                      # 0-9, see /v1/segments
    mode: str = "chat"                                 # "chat" | "voice" (shorter replies)

@app.post("/v1/chat")
def chat(req: ChatRequest):
    try:
        out = get_bot(req.segment).chat([m.model_dump() for m in req.messages], mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return out                                         # {chosen_move, reply, belief, history_len}

@app.get("/v1/segments")
def segments():
    return {i: name for i, name in enumerate(rsa.SEG_NAMES)}

@app.get("/healthz")
def health():
    return {"ok": True, "version": rsa.__version__}
