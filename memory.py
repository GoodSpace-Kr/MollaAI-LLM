from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Callable
import uuid

logger = logging.getLogger("molla.llm.memory")


DEFAULT_USER_STATE: dict[str, Any] = {
    "current_project": None,
    "current_problem": None,
    "current_goal": None,
    "current_topic": None,
    "technical_context": {},
    "preferences": {"language": "English"},
    "decisions": [],
}

ALLOWED_MEMORY_PATHS = {
    "current_project",
    "current_problem",
    "current_goal",
    "current_topic",
    "technical_context.main_llm",
    "technical_context.model_size",
    "technical_context.memory_strategy",
    "preferences.language",
    "preferences.explanation_style",
    "decisions",
}

LOW_VALUE_EXACT = {
    "응",
    "아니",
    "ㅇㅋ",
    "오케이",
    "계속",
    "계속해줘",
    "더 설명",
    "더 설명해줘",
    "예시",
    "예시 보여줘",
    "이해가 안돼",
    "모르겠어",
    "다시 말해줘",
}

HIGH_VALUE_KEYWORDS = [
    "나는",
    "내가",
    "내 프로젝트",
    "프로젝트",
    "목표",
    "문제",
    "결정",
    "바꿀래",
    "안 쓸래",
    "쓸래",
    "사용할래",
    "구현할래",
    "선호",
    "원해",
    "원하는",
    "방식",
    "Qwen",
    "8B",
    "RAG",
    "메모리",
    "Vector DB",
    "지연시간",
    "비동기",
    "동기",
    "Codex",
]


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    id: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MemoryWriteJob:
    user_id: str
    conversation_id: str
    user_input: str
    assistant_answer: str
    recent_messages: list[dict[str, str]]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_qdrant_point_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (TypeError, ValueError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")
    return cleaned or "anonymous"


def _build_heuristic_turn_summary(user_input: str, assistant_answer: str) -> str:
    user_text = " ".join(user_input.split()).strip()
    assistant_text = " ".join(assistant_answer.split()).strip()
    if not user_text:
        return ""
    if len(assistant_text) > 180:
        assistant_text = assistant_text[:177].rstrip() + "..."
    if assistant_text:
        return f"User said: {user_text}\nAssistant replied: {assistant_text}"
    return f"User said: {user_text}"


class UserStateStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or os.getenv("USER_STATE_DIR", "/app/user_states"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load(self, user_id: str) -> dict[str, Any]:
        path = self._path(user_id)
        if not path.exists():
            return deepcopy(DEFAULT_USER_STATE)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("user_state_load_failed user_id=%s path=%s", user_id, path)
            return deepcopy(DEFAULT_USER_STATE)
        return _merge_default_state(data if isinstance(data, dict) else {})

    def save(self, user_id: str, state: dict[str, Any]) -> None:
        path = self._path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, user_id: str) -> Path:
        return self.base_dir / f"{_safe_id(user_id)}.json"


class HashEmbeddingModel:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm <= 0:
            return vector
        return [value / norm for value in vector]


class TransformersEmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.torch = torch
        self.dim = int(getattr(self.model.config, "hidden_size", 384))

    def embed(self, text: str) -> list[float]:
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            masked_embeddings = token_embeddings * attention_mask
            summed = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            vector = summed / counts
            vector = self.torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector[0].cpu().tolist()


class QdrantVectorMemoryStore:
    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str | None = None,
        vector_size: int = 384,
    ) -> None:
        self.url = url or os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "user_memory")
        self.vector_size = vector_size
        self._client = None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            self._models = models
            self._client = QdrantClient(url=self.url)
            self._ensure_collection()
        except Exception:
            self._models = None
            logger.exception("qdrant_init_failed url=%s collection=%s", self.url, self.collection_name)

    @property
    def available(self) -> bool:
        return self._client is not None and self._models is not None

    def search(self, *, embedding: list[float], user_id: str, top_k: int = 3, score_threshold: float = 0.75) -> list[RetrievedMemory]:
        if not self.available:
            return []
        assert self._client is not None
        assert self._models is not None
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key="user_id",
                            match=self._models.MatchValue(value=user_id),
                        )
                    ]
                ),
                limit=top_k,
                score_threshold=score_threshold,
            )
        except Exception:
            logger.exception("qdrant_search_failed user_id=%s collection=%s", user_id, self.collection_name)
            return []

        memories: list[RetrievedMemory] = []
        for result in results:
            payload = result.payload or {}
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            memories.append(
                RetrievedMemory(
                    id=str(result.id),
                    text=text,
                    score=float(result.score),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
            )
        return memories

    def upsert(
        self,
        *,
        memory_id: str,
        user_id: str,
        conversation_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        if not self.available:
            return
        assert self._client is not None
        assert self._models is not None
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "text": text,
            "metadata": metadata,
            "created_at": metadata.get("created_at") or now_iso(),
        }
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    self._models.PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )
        except Exception:
            logger.exception("qdrant_upsert_failed user_id=%s conversation_id=%s", user_id, conversation_id)

    def _ensure_collection(self) -> None:
        assert self._client is not None
        assert self._models is not None
        collections = self._client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._models.VectorParams(
                size=self.vector_size,
                distance=self._models.Distance.COSINE,
            ),
        )


def build_retrieval_query(user_input: str, user_state: dict[str, Any], recent_messages: list[dict[str, str]]) -> str:
    parts = [f"현재 사용자 입력: {user_input}"]
    if user_state.get("current_project"):
        parts.append(f"현재 프로젝트: {user_state['current_project']}")
    if user_state.get("current_problem"):
        parts.append(f"현재 문제: {user_state['current_problem']}")
    if user_state.get("current_goal"):
        parts.append(f"현재 목표: {user_state['current_goal']}")
    if user_state.get("current_topic"):
        parts.append(f"현재 주제: {user_state['current_topic']}")

    technical_context = user_state.get("technical_context") or {}
    if isinstance(technical_context, dict):
        if technical_context.get("main_llm"):
            parts.append(f"메인 LLM: {technical_context['main_llm']}")
        if technical_context.get("memory_strategy"):
            parts.append(f"메모리 전략: {technical_context['memory_strategy']}")

    recent_user_messages = [message["content"] for message in recent_messages if message.get("role") == "user"][-2:]
    if recent_user_messages:
        parts.append("최근 사용자 발화:")
        parts.extend(recent_user_messages)
    return "\n".join(parts)


def format_retrieved_memories(memories: list[RetrievedMemory]) -> str:
    if not memories:
        return "없음"
    lines: list[str] = []
    for index, memory in enumerate(memories, start=1):
        text = memory.text.strip()
        if len(text) > 500:
            text = f"{text[:500]}..."
        if memory.score is None:
            lines.append(f"{index}. {text}")
        else:
            lines.append(f"{index}. {text} (score={memory.score:.2f})")
    return "\n".join(lines)


def format_recent_messages(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "없음"
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) or "없음"


def build_answer_prompt(
    *,
    user_state: dict[str, Any],
    retrieved_memories: list[RetrievedMemory],
    recent_messages: list[dict[str, str]],
    user_input: str,
) -> str:
    return f"""
You are a context-aware spoken English conversation assistant that maintains the user's long-term context.

Important rules:
1. Prioritize the current user input.
2. Use the recent conversation as the second-most important context.
3. Treat User State as the current user/project state.
4. Retrieved Memories are historical records. Use them only when relevant, and remember they can be outdated.
5. If User State conflicts with Retrieved Memories, prefer User State.
6. If the recent conversation conflicts with User State, prefer the recent conversation.
7. If the user says things like "this", "that", "the thing from earlier", or "the project", resolve the reference from Recent Conversation and User State.
8. Ignore Retrieved Memories when they are not relevant to the current question.
9. Do not guess unknown facts.
10. Reply in natural, concise spoken English by default.
11. Use Korean only when the user explicitly asks for Korean or when a brief Korean clarification is necessary.
12. If the user wants to understand a concept, explain it simply first, then make implementation steps concrete.
13. Do not unnecessarily ask again about project context that is already available.
14. Never use emoji, emoticons, decorative symbols, or markdown because this text will be read aloud by TTS.
15. Do not greet the user again unless the current user input is a greeting.
16. Keep the answer short enough for live voice conversation: one or two spoken sentences by default.
17. Text inside [Current User Input] is always the user's speech, not your speech.
18. If the user says "my name is X", remember that X is the user's name. Never say "my name is X"; say "your name is X" or address the user by name.
19. Do not repeat the user's sentence as if you are the user. Respond from the assistant's perspective.

[User State]
{json.dumps(user_state or {}, ensure_ascii=False, indent=2)}

[Retrieved Memories]
{format_retrieved_memories(retrieved_memories)}

[Recent Conversation]
{format_recent_messages(recent_messages)}

[Current User Input]
{user_input}
""".strip()


def should_consider_memory_update(user_input: str, assistant_answer: str | None = None) -> bool:
    text = user_input.strip()
    if len(text) < 8:
        return False
    if text in LOW_VALUE_EXACT:
        return False
    return any(keyword in text for keyword in HIGH_VALUE_KEYWORDS)


def build_memory_writer_prompt(
    *,
    user_state: dict[str, Any],
    recent_messages: list[dict[str, str]],
    user_input: str,
    assistant_answer: str,
) -> str:
    return f"""
너는 대화에서 장기적으로 기억할 정보와 검색용 요약을 생성하는 Memory Writer다.

중요 규칙:
- 사용자가 명확히 말한 사실만 저장한다.
- 추측하지 않는다.
- 단순 맞장구, 일시적 감정, 중복 정보는 저장하지 않는다.
- 현재 user_state와 충돌하는 새 정보가 있으면 update patch를 만든다.
- 기존 정보와 같은 내용이면 patch를 만들지 않는다.
- 결정된 사실과 후보/검토 중인 내용을 구분한다.
- assistant의 설명을 사용자 사실처럼 저장하지 않는다.
- 다만 assistant가 사용자와 합의한 구현 방향은 decisions에 저장할 수 있다.
- 사용자의 선호, 프로젝트, 목표, 문제, 기술적 결정사항은 저장 가치가 높다.
- 출력은 JSON만 한다.

출력 스키마:
{{
  "memory_patch": [
    {{
      "path": "string",
      "action": "set | append | delete | ignore",
      "value": "string | object | array | null",
      "confidence": 0.0,
      "reason": "string"
    }}
  ],
  "turn_summary": "string",
  "should_store_vector": true,
  "importance": 0.0
}}

[Current User State]
{json.dumps(user_state or {}, ensure_ascii=False, indent=2)}

[Recent Messages]
{format_recent_messages(recent_messages)}

[Current User Input]
{user_input}

[Assistant Answer]
{assistant_answer}
""".strip()


def parse_memory_writer_output(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    if "```" in text:
        text = text.replace("```json", "```")
        parts = [part.strip() for part in text.split("```") if part.strip()]
        text = next((part for part in parts if part.startswith("{")), text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("memory_writer_json_parse_failed raw=%r", raw_text[:500])
        return None
    return data if isinstance(data, dict) else None


def apply_memory_patch(user_state: dict[str, Any], memory_patch: list[dict[str, Any]]) -> dict[str, Any]:
    state = _merge_default_state(deepcopy(user_state or {}))
    for patch in memory_patch:
        path = str(patch.get("path", "")).strip()
        action = str(patch.get("action", "")).strip()
        value = patch.get("value")
        try:
            confidence = float(patch.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence < 0.7 or action == "ignore" or path not in ALLOWED_MEMORY_PATHS:
            continue
        if action == "set":
            _set_nested_value(state, path, _truncate_value(value))
        elif action == "append":
            current = _get_nested_value(state, path)
            item = _truncate_value(value)
            if current is None:
                _set_nested_value(state, path, [item])
            elif isinstance(current, list) and item not in current:
                current.append(item)
                _set_nested_value(state, path, current)
        elif action == "delete":
            _delete_nested_value(state, path)

    state["updated_at"] = now_iso()
    return state


class MemoryService:
    def __init__(
        self,
        *,
        state_store: UserStateStore | None = None,
        embedding_model: HashEmbeddingModel | None = None,
        vector_store: QdrantVectorMemoryStore | None = None,
    ) -> None:
        self.state_store = state_store or UserStateStore()
        self.embedding_model = embedding_model or self._load_embedding_model()
        self.vector_store = vector_store or QdrantVectorMemoryStore(vector_size=self.embedding_model.dim)
        self.use_llm_memory_writer = os.getenv("MEMORY_WRITER_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
        self._queue: Queue[tuple[MemoryWriteJob, Callable[[str], str] | None]] = Queue()
        self._worker = Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _load_embedding_model(self):
        if os.getenv("EMBEDDING_MODEL_NAME", "").strip().lower() == "hash":
            return HashEmbeddingModel()
        try:
            model = TransformersEmbeddingModel()
            logger.info("embedding_model_loaded model=%s dim=%s", model.model_name, model.dim)
            return model
        except Exception:
            logger.exception("embedding_model_load_failed fallback=hash")
            return HashEmbeddingModel()

    def load_user_state(self, user_id: str) -> dict[str, Any]:
        return self.state_store.load(user_id)

    def retrieve(self, *, user_id: str, query: str, top_k: int = 3) -> list[RetrievedMemory]:
        embedding = self.embedding_model.embed(query)
        return self.vector_store.search(embedding=embedding, user_id=user_id, top_k=top_k)

    def enqueue_write(self, job: MemoryWriteJob, memory_writer: Callable[[str], str]) -> None:
        if not should_consider_memory_update(job.user_input, job.assistant_answer):
            logger.info("memory_write_skipped user_id=%s conversation_id=%s", job.user_id, job.conversation_id)
            return
        logger.info("memory_write_enqueued user_id=%s conversation_id=%s", job.user_id, job.conversation_id)
        self._queue.put((job, memory_writer if self.use_llm_memory_writer else None))

    def upsert_embedded_points(self, points: list[dict[str, Any]]) -> int:
        count = 0
        for point in points:
            payload = point.get("payload") if isinstance(point.get("payload"), dict) else {}
            user_id = str(payload.get("userId") or payload.get("user_id") or "anonymous")
            conversation_id = str(payload.get("conversationId") or payload.get("conversation_id") or payload.get("phoneNumber") or "external")
            user_text = str(payload.get("userText") or "").strip()
            assistant_text = str(payload.get("assistantText") or "").strip()
            text = "\n".join(part for part in [user_text, assistant_text] if part).strip()
            vector = point.get("vector")
            if not text or not isinstance(vector, list):
                continue
            self.vector_store.upsert(
                memory_id=_coerce_qdrant_point_id(str(point.get("id") or uuid.uuid4())),
                user_id=user_id,
                conversation_id=conversation_id,
                text=text,
                embedding=[float(value) for value in vector],
                metadata={
                    "type": "external_point",
                    "source": "memory_points_endpoint",
                    "created_at": str(payload.get("createdAt") or now_iso()),
                    "audio_key": payload.get("audioKey"),
                },
            )
            count += 1
        return count

    def _run_worker(self) -> None:
        while True:
            job, memory_writer = self._queue.get()
            try:
                self._handle_job(job, memory_writer)
            except Exception:
                logger.exception("memory_write_failed user_id=%s conversation_id=%s", job.user_id, job.conversation_id)
            finally:
                self._queue.task_done()

    def _handle_job(self, job: MemoryWriteJob, memory_writer: Callable[[str], str] | None) -> None:
        logger.info("memory_write_started user_id=%s conversation_id=%s", job.user_id, job.conversation_id)
        if memory_writer is None:
            self._handle_heuristic_job(job)
            return

        user_state = self.state_store.load(job.user_id)
        prompt = build_memory_writer_prompt(
            user_state=user_state,
            recent_messages=job.recent_messages,
            user_input=job.user_input,
            assistant_answer=job.assistant_answer,
        )
        result = parse_memory_writer_output(memory_writer(prompt))
        if result is None:
            return

        patch = result.get("memory_patch") if isinstance(result.get("memory_patch"), list) else []
        updated_state = apply_memory_patch(user_state, patch)
        self.state_store.save(job.user_id, updated_state)

        turn_summary = str(result.get("turn_summary", "")).strip()
        should_store_vector = bool(result.get("should_store_vector", False))
        try:
            importance = float(result.get("importance", 0))
        except (TypeError, ValueError):
            importance = 0.0

        if should_store_vector and turn_summary and importance >= 0.6:
            self.vector_store.upsert(
                memory_id=str(uuid.uuid4()),
                user_id=job.user_id,
                conversation_id=job.conversation_id,
                text=turn_summary,
                embedding=self.embedding_model.embed(turn_summary),
                metadata={
                    "type": "turn_summary",
                    "source": "conversation",
                    "importance": importance,
                    "created_at": now_iso(),
                },
            )
            logger.info("memory_vector_upserted user_id=%s conversation_id=%s importance=%.2f", job.user_id, job.conversation_id, importance)
        logger.info("memory_write_done user_id=%s conversation_id=%s patches=%s", job.user_id, job.conversation_id, len(patch))

    def _handle_heuristic_job(self, job: MemoryWriteJob) -> None:
        summary = _build_heuristic_turn_summary(job.user_input, job.assistant_answer)
        if not summary:
            return
        self.vector_store.upsert(
            memory_id=str(uuid.uuid4()),
            user_id=job.user_id,
            conversation_id=job.conversation_id,
            text=summary,
            embedding=self.embedding_model.embed(summary),
            metadata={
                "type": "turn_summary",
                "source": "conversation_heuristic",
                "created_at": now_iso(),
            },
        )
        logger.info("memory_write_done user_id=%s conversation_id=%s mode=heuristic", job.user_id, job.conversation_id)


def _merge_default_state(data: dict[str, Any]) -> dict[str, Any]:
    state = deepcopy(DEFAULT_USER_STATE)
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            merged = deepcopy(state[key])
            merged.update(value)
            state[key] = merged
        else:
            state[key] = value
    return state


def _get_nested_value(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested_value(data: dict[str, Any], path: str, value: Any) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _delete_nested_value(data: dict[str, Any], path: str) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(parts[-1], None)


def _truncate_value(value: Any, limit: int = 500) -> Any:
    if isinstance(value, str):
        return value[:limit]
    return value
