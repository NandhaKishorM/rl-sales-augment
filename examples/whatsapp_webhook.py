"""WhatsApp Business Cloud API webhook -> the RL sales agent.

    pip install "rl-sales-augment[gemini,api]" httpx
    WA_TOKEN=... WA_PHONE_ID=... WA_VERIFY_TOKEN=anything GCP_PROJECT=... \\
        uvicorn whatsapp_webhook:app --host 0.0.0.0 --port 8000

Point your Meta app's webhook at https://<host>/webhook with the same verify token.
Each WhatsApp sender gets their own agent session (belief + history survive across messages),
turns are logged to JSONL, and any escalation flag pauses the bot for that customer so a
human can take over. Manglish/Hindi/Malayalam customers get replies in their own language
automatically (see the multilingual section of the README).
"""
import os
import httpx
from fastapi import FastAPI, Request, Response
import rl_sales_augment as rsa

WA_TOKEN = os.environ.get("WA_TOKEN", "")
WA_PHONE_ID = os.environ.get("WA_PHONE_ID", "")
VERIFY_TOKEN = os.environ.get("WA_VERIFY_TOKEN", "verify-me")
COMPANY_CTX = os.environ.get("RSA_COMPANY_CTX", "")

gen = rsa.providers.gemini_vertex()                      # or openai_chat()/anthropic_chat()
app = FastAPI(title="rl-sales-augment WhatsApp bridge")
_sessions: dict = {}                                     # wa_id -> agent
_paused: set = set()                                     # escalated customers wait for a human


def get_bot(wa_id):
    if wa_id not in _sessions:
        _sessions[wa_id] = rsa.load_agent(gen, company_ctx=COMPANY_CTX,
                                          log_path="whatsapp_turns.jsonl")
    return _sessions[wa_id]


async def send_whatsapp(to, text):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v21.0/{WA_PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            json={"messaging_product": "whatsapp", "to": to,
                  "type": "text", "text": {"body": text[:4000]}})


@app.get("/webhook")
def verify(request: Request):
    q = request.query_params
    if q.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(q.get("hub.challenge", ""), media_type="text/plain")
    return Response(status_code=403)


@app.post("/webhook")
async def incoming(request: Request):
    data = await request.json()
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []) or []:
                if msg.get("type") != "text":
                    continue
                wa_id, text = msg["from"], msg["text"]["body"]
                if wa_id in _paused:
                    continue                              # a human owns this thread now
                out = get_bot(wa_id).reply(text)
                await send_whatsapp(wa_id, out["reply"])
                if out.get("escalate"):
                    _paused.add(wa_id)                    # stop the bot, notify your team
                    print(f"ESCALATE {wa_id}: {out.get('escalate_reason')}")
    return {"ok": True}


@app.post("/resume/{wa_id}")                             # human done -> bot resumes
def resume(wa_id: str):
    _paused.discard(wa_id)
    return {"resumed": wa_id}
