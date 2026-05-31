import json
import logging
import time
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from llm import QwenChat
from memory import (
    MemoryService,
    MemoryWriteJob,
    build_answer_prompt,
    build_retrieval_query,
)


logger = logging.getLogger("molla.llm")
app = FastAPI(title="molla-llm")
chat = QwenChat()
memory_service = MemoryService()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    conversation_id: str = "default"
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str


class MemoryPointPayload(BaseModel):
    userId: str | None = None
    phoneNumber: str | None = None
    userText: str | None = None
    assistantText: str | None = None
    createdAt: str | None = None
    audioKey: str | None = None


class MemoryPoint(BaseModel):
    id: str
    vector: list[float] = Field(min_length=1)
    payload: MemoryPointPayload


class MemoryPointsRequest(BaseModel):
    points: list[MemoryPoint] = Field(min_length=1)


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
    prompt = build_contextual_prompt(payload=payload, history=history)
    answer = chat.ask(prompt)
    enqueue_memory_write(payload=payload, history=history, answer=answer)
    return ChatResponse(answer=answer)


def token_event_stream(payload: ChatRequest, history: list[dict[str, str]]) -> Iterator[str]:
    prompt = build_contextual_prompt(payload=payload, history=history)
    chunks: list[str] = []
    for token in chat.stream_answer(prompt):
        chunks.append(token)
        event_payload = json.dumps({"token": token}, ensure_ascii=False)
        yield f"data: {event_payload}\n\n"
    enqueue_memory_write(payload=payload, history=history, answer="".join(chunks).strip())
    yield "data: [DONE]\n\n"


@app.post("/chat/tokens")
def chat_token_stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    history = [message.model_dump(mode="python") for message in payload.history]
    return StreamingResponse(token_event_stream(payload, history=history), media_type="text/event-stream")


@app.post("/chat/stream", response_model=ChatResponse)
async def chat_stream_endpoint(request: Request) -> ChatResponse:
    query = await read_streamed_text(request)
    payload = ChatRequest(query=query)
    prompt = build_contextual_prompt(payload=payload, history=[])
    answer = chat.ask(prompt)
    enqueue_memory_write(payload=payload, history=[], answer=answer)
    return ChatResponse(answer=answer)


@app.post("/memory/points")
def upsert_memory_points(payload: MemoryPointsRequest) -> dict[str, int | str]:
    count = memory_service.upsert_embedded_points(payload.model_dump(mode="python")["points"])
    return {"status": "ok", "count": count}


def build_contextual_prompt(*, payload: ChatRequest, history: list[dict[str, str]]) -> str:
    started_at = time.perf_counter()
    user_state = memory_service.load_user_state(payload.user_id)
    retrieval_query = build_retrieval_query(payload.query, user_state, history)
    memories = memory_service.retrieve(user_id=payload.user_id, query=retrieval_query, top_k=3)
    prompt = build_answer_prompt(
        user_state=user_state,
        retrieved_memories=memories,
        recent_messages=history,
        user_input=payload.query,
    )
    logger.info(
        "llm_context_built user_id=%s conversation_id=%s retrieved=%s prompt_chars=%s elapsed_ms=%s",
        payload.user_id,
        payload.conversation_id,
        len(memories),
        len(prompt),
        int((time.perf_counter() - started_at) * 1000),
    )
    return prompt


def enqueue_memory_write(*, payload: ChatRequest, history: list[dict[str, str]], answer: str) -> None:
    if not answer.strip():
        return
    memory_service.enqueue_write(
        MemoryWriteJob(
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            user_input=payload.query,
            assistant_answer=answer,
            recent_messages=[*history, {"role": "user", "content": payload.query}, {"role": "assistant", "content": answer}],
        ),
        memory_writer=chat.generate_memory_json,
    )
