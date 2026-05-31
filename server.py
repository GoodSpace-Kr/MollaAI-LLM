import json
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from llm import QwenChat


app = FastAPI(title="molla-llm")
chat = QwenChat()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str


async def read_streamed_text(request: Request) -> str:
    chunks: list[str] = []

    async for chunk in request.stream():
        if not chunk:
            continue
        chunks.append(chunk.decode("utf-8"))

    raw_text = "".join(chunks).strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty streamed body")

    if "data:" in raw_text:
        sse_chunks: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            sse_chunks.append(payload)
        if sse_chunks:
            raw_text = " ".join(sse_chunks).strip()

    ndjson_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if ndjson_lines and all(line.startswith("{") and line.endswith("}") for line in ndjson_lines):
        text_parts: list[str] = []
        for line in ndjson_lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                text_parts = []
                break

            if isinstance(data, dict):
                text = data.get("text") or data.get("query") or data.get("partial")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())

        if text_parts:
            raw_text = " ".join(text_parts).strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="No text extracted from stream")

    return raw_text


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    history = [message.model_dump(mode="python") for message in payload.history]
    return ChatResponse(answer=chat.ask(payload.query, history=history))


def token_event_stream(query: str, history: list[dict[str, str]] | None = None) -> Iterator[str]:
    for token in chat.stream_answer(query, history=history):
        payload = json.dumps({"token": token}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/tokens")
def chat_token_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    history = [message.model_dump(mode="python") for message in payload.history]
    return StreamingResponse(token_event_stream(payload.query, history=history), media_type="text/event-stream")


@app.post("/chat/stream", response_model=ChatResponse)
async def chat_stream_endpoint(request: Request) -> ChatResponse:
    query = await read_streamed_text(request)
    return ChatResponse(answer=chat.ask(query))
