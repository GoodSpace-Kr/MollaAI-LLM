import os
import traceback
import warnings
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

warnings.filterwarnings("ignore")

TOK_MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "ssaann/eng_conversation_sft"

gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

summary_gen_config = GenerationConfig(
    max_new_tokens=180,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
)

BASE_SYSTEM_PROMPT = (
    "You are an English conversation teacher on a live phone call. "
    "Reply briefly in plain spoken English, only as the teacher, "
    "and never generate the student's side of the conversation."
)

SUMMARY_SYSTEM_PROMPT = (
    "You summarize a conversation for future turns. "
    "Keep only stable facts, user preferences, open questions, corrections, and important context. "
    "Return plain text only. No bullets, no markdown, no extra commentary."
)


class QwenChat:
    def __init__(
        self,
        model_name=MODEL_NAME,
        tok_model=TOK_MODEL_NAME,
        cache_dir=None,
        token_limit: int = 3000,
        summarize_threshold: float = 0.7,
        recent_message_count: int = 6,
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ):
        print("🔧 LLM 모델 로딩 중...")

        cache_dir = cache_dir or os.getenv("HF_HOME")

        self.tokenizer = AutoTokenizer.from_pretrained(tok_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )

        self.token_limit = token_limit
        self.summarize_threshold = summarize_threshold
        self.recent_message_count = max(2, recent_message_count)
        self.system_prompt = system_prompt

        self.history: list[dict[str, str]] = []
        self.summary: str = ""

        print("✅ LLM 준비 완료")

    def reset_memory(self) -> None:
        self.history.clear()
        self.summary = ""

    def _build_system_prompt(self) -> str:
        if not self.summary.strip():
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            f"Conversation summary so far:\n{self.summary.strip()}"
        )

    def _build_messages(self, query: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": query})
        return messages

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            return inputs
        return {key: value.to(model_device) for key, value in inputs.items()}

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        prompt = self._render_prompt(messages)
        return int(self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1])

    def _generate_text(self, prompt: str, generation_config: GenerationConfig) -> str:
        inputs = self._tokenize_prompt(prompt)
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _summarize_history(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return self.summary.strip()

        transcript_lines: list[str] = []
        for message in messages:
            role = message["role"].capitalize()
            transcript_lines.append(f"{role}: {message['content']}")
        transcript = "\n".join(transcript_lines)

        summary_messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Update the running conversation summary for the next turn.\n\n"
                    f"Current summary:\n{self.summary.strip() or 'None'}\n\n"
                    f"New conversation turns:\n{transcript}\n\n"
                    "Write a compact updated summary."
                ),
            },
        ]
        prompt = self._render_prompt(summary_messages)
        summary = self._generate_text(prompt, summary_gen_config)
        return summary or self.summary.strip()

    def _maybe_summarize_memory(self, upcoming_query: str | None = None) -> None:
        candidate_messages = self._build_messages(upcoming_query or "")
        token_count = self._estimate_tokens(candidate_messages)
        threshold = int(self.token_limit * self.summarize_threshold)

        if token_count <= threshold or len(self.history) <= self.recent_message_count:
            return

        old_messages = self.history[:-self.recent_message_count]
        recent_messages = self.history[-self.recent_message_count :]
        updated_summary = self._summarize_history(old_messages)

        self.summary = updated_summary.strip()
        self.history = recent_messages

    def get_prompt(self, query: str):
        self._maybe_summarize_memory(query)
        messages = self._build_messages(query)
        prompt = self._render_prompt(messages)
        return query, prompt

    def answer_extraction(self, response: str) -> str:
        try:
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant", 1)[1]

            if "</think>" in response:
                think_end = response.find("</think>") + len("</think>")
                response = response[think_end:]

            return response.strip()

        except Exception:
            print("Error in answer_extraction")
            print(traceback.format_exc())
            return response

    def stream_answer(self, query: str):
        try:
            _, prompt = self.get_prompt(query)
            inputs = self._tokenize_prompt(prompt)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "generation_config": gen_config,
                "streamer": streamer,
            }

            worker = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
                daemon=True,
            )
            worker.start()

            for chunk in streamer:
                if chunk:
                    yield chunk

            worker.join()
        except Exception:
            print("Error in stream_answer()")
            print(traceback.format_exc())
            yield "오류가 발생했습니다."

    def ask(self, query: str) -> str:
        try:
            answer = self.answer_extraction("".join(self.stream_answer(query)))
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": answer})
            self._maybe_summarize_memory()
            return answer

        except Exception:
            print("Error in ask()")
            print(traceback.format_exc())
            return "오류가 발생했습니다."

    def chat(self, query: str) -> str:
        return self.ask(query)